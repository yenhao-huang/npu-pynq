"""Verilator lint backend.

Verilator's diagnostics look like:

    %Warning-WIDTHEXPAND: src/hw/rtl/x.sv:41:23: Operator ASSIGN expects 32 ...
                                                : ... note: In instance ...
    %Error: src/hw/rtl/x.sv:12:1: syntax error, unexpected ';'

The first line of each record carries everything `LintOut` needs; the indented
continuation lines are notes that repeat what the agent already has, so they
are folded away rather than emitted as separate issues.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...process import run as run_process
from ...registry import backend
from . import Issue, LintIn, LintOut

# %Error-CODE: file:line:col: message   (code, file, line and col all optional)
DIAG = re.compile(
    r"^%(?P<kind>Error|Warning)(?:-(?P<code>[A-Z0-9_]+))?:\s*"
    r"(?:(?P<file>[^:\s][^:]*):(?P<line>\d+):(?:(?P<col>\d+):)?\s*)?"
    r"(?P<message>.*)$"
)


def parse(text: str, root: Path | None = None) -> list[Issue]:
    issues: list[Issue] = []
    for raw in text.splitlines():
        match = DIAG.match(raw.strip())
        if not match:
            continue
        code = match.group("code")
        message = match.group("message").strip()
        # "%Error: Exiting due to 3 warning(s)" is a tally of the diagnostics
        # already reported, not a diagnostic. Other file-less errors -- a
        # missing module, say -- are real and are kept.
        if message.startswith("Exiting due to"):
            continue
        path = match.group("file") or "<unknown>"
        if root is not None:
            try:
                path = str(Path(path).resolve().relative_to(root))
            except (ValueError, OSError):
                pass
        issues.append(
            Issue(
                file=path,
                line=int(match.group("line")) if match.group("line") else None,
                column=int(match.group("col")) if match.group("col") else None,
                severity="error" if match.group("kind") == "Error" else "warning",
                code=code,
                message=message,
            )
        )
    return issues


@backend("lint", "verilator", requires="verilator", version_cmd=["verilator", "--version"])
class VerilatorLint:
    def lint(self, params: LintIn, ctx) -> LintOut:
        argv = ["verilator", "--lint-only"]
        if params.strict:
            argv += ["-Wall"]
        # Without this, a clean-but-warned design exits non-zero and the
        # warnings after the first are never reported.
        argv += ["-Wno-fatal"]
        for directory in params.include_dirs:
            argv += ["-I" + directory]
        for define in params.defines:
            argv += ["-D" + define]
        if params.top:
            argv += ["--top-module", params.top]
        argv += list(params.files)

        result = run_process(argv, log_path=ctx.run.artifacts / "lint.log",
                             cwd=ctx.cwd, timeout_s=600)
        issues = parse(result.text(), ctx.cwd)
        errors = sum(1 for i in issues if i.severity == "error")
        return LintOut(
            ok=errors == 0 and result.exit_code == 0,
            error_count=errors,
            warning_count=sum(1 for i in issues if i.severity == "warning"),
            issues=issues,
            backend="verilator",
            backend_version=ctx.backend_version,
        )
