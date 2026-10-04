from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from tpugen_types import FlowError

from .base import LLMBackend, clean_vh
from .api import SYSTEM


class CLIAgentBackend(LLMBackend):
    """A local coding-agent CLI driven in one-shot, non-interactive mode.

    The agent is given no reason to touch the filesystem: the prompt is the
    whole task and the answer is the header text. Each backend still pins the
    CLI to its read-only / no-tool mode so a stray tool call cannot modify the
    working tree.
    """

    executable: str = ""
    default_model: str | None = None

    def __init__(self, model: str | None = None, timeout: int = 600):
        if not self.executable:
            raise FlowError("CLIAgentBackend subclass must set `executable`")
        self.path = shutil.which(self.executable)
        if not self.path:
            raise FlowError(
                f"{self.executable!r} CLI not found on PATH; install it or pick "
                f"another backend"
            )
        self.model = model or self.default_model
        self.timeout = timeout

    # --- subclass hooks -------------------------------------------------
    def _command(self, prompt: str, out_file: Path) -> list[str]:
        raise NotImplementedError

    def _read_reply(self, stdout: str, out_file: Path) -> str:
        return stdout

    # --- driver ---------------------------------------------------------
    def generate(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix=f"tpugen-{self.name}-") as tmp:
            out_file = Path(tmp) / "reply.txt"
            cmd = self._command(prompt, out_file)
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    env={**os.environ, "NO_COLOR": "1"},
                )
            except subprocess.TimeoutExpired as exc:
                raise FlowError(
                    f"{self.name} CLI timed out after {self.timeout}s"
                ) from exc
            if proc.returncode != 0:
                raise FlowError(
                    f"{self.name} CLI exited {proc.returncode}:\n"
                    + (proc.stderr or proc.stdout)[-2000:]
                )
            reply = self._read_reply(proc.stdout, out_file)
        if not reply.strip():
            raise FlowError(f"{self.name} CLI returned an empty reply")
        return clean_vh(reply)


class ClaudeCLIBackend(CLIAgentBackend):
    """Claude Code in print mode (`claude -p`)."""

    name = "claude"
    executable = "claude"
    default_model = "claude-sonnet-5"

    def _command(self, prompt: str, out_file: Path) -> list[str]:
        cmd = [
            self.path, "-p", prompt,
            "--output-format", "text",
            "--append-system-prompt", SYSTEM,
            "--permission-mode", "plan",
            "--disallowed-tools", "Bash", "Edit", "Write", "WebFetch", "WebSearch",
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd


class CodexCLIBackend(CLIAgentBackend):
    """OpenAI Codex CLI in non-interactive mode (`codex exec`)."""

    name = "codex"
    executable = "codex"
    default_model = None  # whatever ~/.codex/config.toml selects

    def _command(self, prompt: str, out_file: Path) -> list[str]:
        cmd = [
            self.path, "exec",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color", "never",
            "--output-last-message", str(out_file),
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd + [f"{SYSTEM}\n\n{prompt}"]

    def _read_reply(self, stdout: str, out_file: Path) -> str:
        # `codex exec` prints its session log to stdout; the answer itself is
        # only reliable from --output-last-message.
        if out_file.exists():
            return out_file.read_text(encoding="utf-8")
        return stdout
