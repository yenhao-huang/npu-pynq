import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.config import Config
from src.agent.codex import CodexModel, normalize
from src.agent.context import Context
from src.agent.parse import parse_stream, summarize
from src.agent.rounds import run_rounds
from src.logging.recorder import Recorder


def codex_stream(thread="t-1", fail=None):
    events = [{"type": "thread.started", "thread_id": thread}, {"type": "turn.started"},
              {"type": "item.completed", "item": {"type": "agent_message", "text": "plan"}},
              {"type": "item.completed", "item": {"type": "file_change",
                                                  "changes": [{"path": "/w/solution.v", "kind": "add"}]}},
              {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "ic", "tool": "ppa",
                                                  "arguments": {"top": "x"}, "status": "completed"}},
              {"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}]
    if fail:
        events.append({"type": "turn.failed", "error": {"message": fail}})
    else:
        events.append({"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60,
                                                            "output_tokens": 7, "reasoning_output_tokens": 3}})
    return "\n".join(json.dumps(e) for e in events)


class CodexTests(unittest.TestCase):
    def test_normalize_feeds_existing_parser(self):
        stdout = codex_stream() + "\n" + normalize(codex_stream(), 0)
        calls, result, _, last, iterations = parse_stream(stdout)
        self.assertEqual(calls, {"Write": 1, "ppa": 1})
        self.assertEqual(iterations, 1)
        self.assertEqual(last, "final")
        self.assertFalse(result["is_error"])
        summary = summarize([stdout])
        self.assertEqual(summary["usage"]["input_tokens"], 40)
        self.assertEqual(summary["usage"]["cache_read_input_tokens"], 60)
        self.assertEqual(summary["cost_usd"], 0)

    def test_failed_turn_is_an_error(self):
        _, result, *_ = parse_stream(normalize(codex_stream(fail="usage limit reached"), 1))
        self.assertTrue(result["is_error"])
        self.assertEqual(result["result"], "usage limit reached")
        _, result, *_ = parse_stream(normalize("", 1))
        self.assertTrue(result["is_error"])

    def test_command_tools_and_resume(self):
        model = CodexModel(Config(model="gpt-6-sol", rounds=3))
        work = Path("/w")
        cmd, _ = model.command("p", "notools", work, ["--no-session-persistence"])
        self.assertIn("--ephemeral", cmd)
        self.assertNotIn("mcp_servers.ic.disabled_tools=[\"show_wave\"]", cmd)
        self.assertEqual(cmd[cmd.index("--disable") + 1], "shell_tool")
        self.assertIn('model_reasoning_effort="medium"', cmd)
        cmd, _ = model.command("p", "tools", work, ["--session-id", "s"])
        self.assertIn('mcp_servers.ic.disabled_tools=["show_wave"]', cmd)
        self.assertNotIn("resume", cmd)
        with self.assertRaises(RuntimeError):
            model.command("p", "tools", work, ["--resume", "s"])
        model.threads["s"] = "t-9"
        cmd, _ = model.command("next", "tools", work, ["--resume", "s"])
        self.assertEqual(cmd[-3:], ["resume", "t-9", "next"])

    def test_multi_round_resumes_recorded_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            model = CodexModel(Config(model="gpt-6-sol", rounds=2))
            done = subprocess.CompletedProcess([], 0, codex_stream("thread-a"), "")
            with patch("src.agent.codex.subprocess.run", return_value=done) as run:
                context = Context("spec", "tools", 2)
                records = run_rounds(context, model, work, Recorder(work / "log"))
            self.assertEqual([r["error"] for r in records], [None, None])
            second = run.call_args_list[1].args[0]
            self.assertEqual(second[-3], "resume")
            self.assertEqual(second[-2], "thread-a")


class ShellModeTests(unittest.TestCase):
    def test_codex_shell_keeps_shell_tool_and_pins_path(self):
        cmd, _ = CodexModel(Config(model="gpt-6-sol", shell=True)).command("p", "notools", Path("/w"), ["--no-session-persistence"])
        disabled = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--disable"]
        self.assertNotIn("shell_tool", disabled)
        self.assertNotIn("unified_exec", disabled)
        self.assertIn(f'shell_environment_policy.set={{PATH="{Config().shell_path}"}}', cmd)
        self.assertNotIn(".venv", Config().shell_path)

    def test_claude_shell_adds_bash_to_both_arms_without_ic_on_path(self):
        from src.agent.llm import ClaudeModel
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            done = subprocess.CompletedProcess([], 0, "{}", "")
            for arm, rounds in (("notools", 1), ("tools", 1), ("notools", 3)):
                with patch("src.agent.llm.subprocess.run", return_value=done) as run:
                    ClaudeModel(Config(shell=True, rounds=rounds)).generate("p", arm, work, ["--no-session-persistence"])
                cmd, env = run.call_args.args[0], run.call_args.kwargs["env"]
                tools = cmd[cmd.index("--tools") + 1].split(",")
                self.assertIn("Bash", tools)
                self.assertEqual(env["PATH"], Config().shell_path)
                self.assertEqual("mcp__ic" in cmd, arm == "tools")
            with patch("src.agent.llm.subprocess.run", return_value=done) as run:
                ClaudeModel(Config()).generate("p", "notools", work, ["--no-session-persistence"])
            self.assertNotIn("Bash", run.call_args.args[0][run.call_args.args[0].index("--tools") + 1])


if __name__ == "__main__":
    unittest.main()
