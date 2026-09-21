"""Subprocess helper.

stdout and stderr go straight to files under the run directory. A Verilator
build log or a Vivado transcript is megabytes; none of it belongs in memory,
and none of it belongs in an agent's context.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    argv: list[str]
    exit_code: int
    duration_s: float
    log_path: Path
    timed_out: bool = False

    def text(self, limit_bytes: int = 4_000_000) -> str:
        """Read the captured log back, capped so a runaway log cannot blow up."""
        if not self.log_path.exists():
            return ""
        data = self.log_path.read_bytes()[:limit_bytes]
        return data.decode("utf-8", errors="replace")


def run(
    argv: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    timeout_s: float | None = None,
    env: dict | None = None,
    append: bool = False,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    mode = "ab" if append else "wb"
    with open(log_path, mode) as sink:
        sink.write(f"$ {' '.join(argv)}\n".encode())
        sink.flush()
        try:
            proc = subprocess.Popen(
                argv, stdout=sink, stderr=subprocess.STDOUT, cwd=cwd, env=env
            )
        except FileNotFoundError as exc:
            sink.write(f"{exc}\n".encode())
            return CommandResult(argv, 127, time.monotonic() - started, log_path)
        try:
            exit_code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                exit_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = proc.wait()
            sink.write(f"\n[ic] timed out after {timeout_s}s\n".encode())
    return CommandResult(
        argv, exit_code, time.monotonic() - started, log_path, timed_out
    )
