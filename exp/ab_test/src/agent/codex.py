"""Codex CLI adapter (e.g. gpt-6-sol) with the same generate() contract as ClaudeModel.

The raw Codex JSONL is kept verbatim; Claude-style ``assistant`` and ``result`` events are
appended (marked ``"synthesized": "codex"``) so parse.py and rounds.py work unchanged.
"""
import json
import os
import subprocess
import threading
import time
from .llm import ModelResponse

# Codex tools that have no counterpart in the Claude arms. Shell execution is disabled so
# both backends edit files only through file tools (apply_patch here, Write/Edit there).
DISABLED_FEATURES = ["shell_tool", "unified_exec", "apps", "plugins", "remote_plugin",
                     "browser_use", "browser_use_external", "computer_use", "multi_agent",
                     "goals", "sleep_tool", "image_generation", "view_image", "skill_search",
                     "tool_suggest"]


def toml_table(values):
    return "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in values.items()) + "}"


def normalize(stdout, returncode):
    """Translate Codex events into the Claude stream-json events parse.py reads."""
    events, last_text, turns, failure = [], "", 0, None
    usage, completed = {}, False
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind, item = ev.get("type"), ev.get("item") or {}
        if kind == "item.completed":
            block = None
            if item.get("type") == "agent_message":
                last_text = item.get("text", "")
                block = {"type": "text", "text": last_text}
            elif item.get("type") == "mcp_tool_call":
                block = {"type": "tool_use", "name": f"mcp__{item.get('server')}__{item.get('tool')}",
                         "input": item.get("arguments") or {}}
            elif item.get("type") == "file_change":
                paths = [c.get("path", "") for c in item.get("changes", [])]
                block = {"type": "tool_use", "name": "Write", "input": {"file_path": " ".join(paths)}}
            elif item.get("type") == "command_execution":
                block = {"type": "tool_use", "name": "Bash", "input": {"command": item.get("command")}}
            elif item.get("type") not in (None, "reasoning"):
                block = {"type": "tool_use", "name": item["type"], "input": {}}
            if block:
                turns += block["type"] == "tool_use"
                events.append({"type": "assistant", "synthesized": "codex", "message": {"content": [block]}})
        elif kind == "turn.completed":
            completed = True
            for k, v in (ev.get("usage") or {}).items():
                usage[k] = usage.get(k, 0) + v
        elif kind == "turn.failed":
            failure = (ev.get("error") or {}).get("message") or "turn failed"
        elif kind == "error" and ev.get("message"):
            failure = failure or ev["message"]
    if not completed and failure is None and returncode not in (0, None):
        failure = "codex process failed"
    cached = usage.get("cached_input_tokens", 0)
    events.append({"type": "result", "synthesized": "codex",
                   "is_error": failure is not None or not completed,
                   "result": failure if failure is not None else last_text,
                   "num_turns": turns, "total_cost_usd": None,
                   "usage": {"input_tokens": usage.get("input_tokens", 0) - cached,
                             "cache_read_input_tokens": cached,
                             "output_tokens": usage.get("output_tokens", 0),
                             "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0)}})
    return "\n".join(json.dumps(e) for e in events)


class CodexModel:
    def __init__(self, config):
        self.config = config
        self.threads = {}  # harness session id -> Codex thread id, for multi-round resume
        self.lock = threading.Lock()

    def command(self, prompt, arm, work, session_flags):
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "--ignore-user-config",
               "--ignore-rules", "-m", self.config.model,
               "-c", f'model_reasoning_effort="{self.config.effort}"',
               "-s", "workspace-write", "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
               "-C", str(work)]
        for feature in DISABLED_FEATURES:
            if self.config.shell and feature in ("shell_tool", "unified_exec"):
                continue
            cmd += ["--disable", feature]
        if self.config.shell:
            cmd += ["-c", 'shell_environment_policy.inherit="core"',
                    "-c", f'shell_environment_policy.set={{PATH="{self.config.shell_path}"}}']
        if arm != "notools":
            env = dict(self.config.tool_env, IC_ROOT=str(work / ".ic"))
            cmd += ["-c", f'mcp_servers.ic.command="{self.config.venv / "ic-mcp"}"',
                    "-c", f"mcp_servers.ic.env={toml_table(env)}",
                    "-c", 'mcp_servers.ic.disabled_tools=["show_wave"]',
                    "-c", 'mcp_servers.ic.default_tools_approval_mode="approve"',
                    "-c", f"mcp_servers.ic.tool_timeout_sec={self.config.timeout}",
                    "-c", "mcp_servers.ic.startup_timeout_sec=60"]
        if session_flags == ["--no-session-persistence"]:
            return cmd + ["--ephemeral", prompt], None
        flag, session = session_flags
        if flag == "--resume":
            with self.lock:
                thread = self.threads.get(session)
            if thread is None:
                raise RuntimeError(f"no Codex thread recorded for session {session}")
            return cmd + ["resume", thread, prompt], session
        return cmd + [prompt], session

    def generate(self, prompt, arm, work, session_flags):
        cmd, session = self.command(prompt, arm, work, session_flags)
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_") and k != "CLAUDECODE"}
        t0 = time.time()
        returncode, stderr, timed_out = None, "", False
        try:
            p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True,
                               timeout=self.config.timeout, stdin=subprocess.DEVNULL)
            raw, returncode, stderr = p.stdout, p.returncode, p.stderr
        except subprocess.TimeoutExpired as e:
            raw = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            timed_out = True
        if session is not None:
            for line in raw.splitlines():
                if '"thread.started"' in line:
                    with self.lock:
                        self.threads.setdefault(session, json.loads(line).get("thread_id"))
        stdout = raw.rstrip("\n") + ("\n" if raw.strip() else "") + normalize(raw, returncode)
        return ModelResponse(stdout, time.time() - t0, timed_out, returncode, stderr)
