"""Claude CLI adapter. Round orchestration depends only on generate()."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

@dataclass
class ModelResponse:
    stdout: str
    wall_s: float
    timed_out: bool = False
    returncode: int | None = 0
    stderr: str = ""

class ClaudeModel:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt, arm, work, session_flags):
        cmd = [shutil.which("claude") or "claude", "-p", prompt, "--model", self.config.model,
               "--output-format", "stream-json", "--verbose",
               "--effort", self.config.effort,
               "--max-budget-usd", self.config.budget,
               "--setting-sources", "", "--disable-slash-commands",
               "--strict-mcp-config", "--permission-mode", "acceptEdits",
               "--disallowedTools", f"Read({Path.home()}/.ab_private/**)", f"Read({self.config.problems}/**)", "mcp__ic__show_wave"]
        cmd += session_flags
        if arm == "notools":
            allowed = ["Write", "Read"] if self.config.rounds > 1 else ["Write"]
            allowed += ["Bash"] if self.config.shell else []
            cmd += ["--tools", ",".join(allowed), "--allowedTools", *allowed]
        else:
            cfg = work / "mcp.json"
            cfg.write_text(json.dumps({"mcpServers": {"ic": {
                "command": str(self.config.venv / "ic-mcp"), "args": [],
                "env": dict(self.config.tool_env, IC_ROOT=str(work / ".ic"))}}}))
            files = ["Write", "Read", "Edit"] + (["Bash"] if self.config.shell else [])
            cmd += ["--mcp-config", str(cfg), "--tools", ",".join(files),
                    "--allowedTools", *files, "mcp__ic"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_") and k != "CLAUDECODE"}
        if self.config.shell:
            env["PATH"] = self.config.shell_path  # Bash inherits this; the ic venv stays off it
        t0 = time.time()
        returncode, stderr = None, ""
        try:
            p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=self.config.timeout)
            stdout, timed_out = p.stdout, False
            returncode, stderr = p.returncode, p.stderr
        except subprocess.TimeoutExpired as e:
            stdout, timed_out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), True
        return ModelResponse(stdout, time.time() - t0, timed_out, returncode, stderr)
