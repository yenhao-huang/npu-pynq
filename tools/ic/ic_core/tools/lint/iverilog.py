"""Icarus Verilog lint backend.

A second backend earns its place by proving the contract holds: adding it
touched this file only. Icarus is stricter than Verilator about some legacy
constructs and laxer about width mismatches, so it is a useful cross-check --
not a replacement for the default.

Icarus diagnostics are `file:line: message`, with severity carried in the
prose rather than a field, and there are no rule codes.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...process import run as run_process
from ...registry import backend
from . import Issue, LintIn, LintOut

DIAG = re.compile(r"^(?P<file>[^:\s][^:]*):(?P<line>\d+):\s*(?P<message>.*)$")


def parse(text: str, root: Path | None = None) -> list[Issue]:
    issues: list[Issue] = []
    for raw in text.splitlines():
        match = DIAG.match(raw.strip())
        if not match:
            continue
        message = match.group("message").strip()
        lowered = message.lower()
        if lowered.startswith("warning"):
            severity = "warning"
        elif lowered.startswith(("error", "syntax error")) or "error:" in lowered:
            severity = "error"
        else:
            severity = "error"
        path = match.group("file")
        if root is not None:
            try:
                path = str(Path(path).resolve().relative_to(root))
            except (ValueError, OSError):
                pass
        issues.append(
            Issue(
                file=path,
                line=int(match.group("line")),
                column=None,
                severity=severity,
                code=None,
                message=message,
            )
        )
    return issues


@backend("lint", "iverilog", requires="iverilog", version_cmd=["iverilog", "-V"])
class IcarusLint:
    def lint(self, params: LintIn, ctx) -> LintOut:
        argv = ["iverilog", "-g2012", "-t", "null"]
        if params.strict:
            argv += ["-Wall"]
        for directory in params.include_dirs:
            argv += ["-I" + directory]
        for define in params.defines:
            argv += ["-D" + define]
        if params.top:
            argv += ["-s", params.top]
        argv += list(params.files)

        result = run_process(argv, log_path=ctx.run.artifacts / "lint.log",
                             cwd=ctx.cwd, timeout_s=600)
        issues = parse(result.text(), ctx.cwd)
        errors = sum(1 for i in issues if i.severity == "error")
        return LintOut(
            ok=result.exit_code == 0,
            error_count=errors,
            warning_count=sum(1 for i in issues if i.severity == "warning"),
            issues=issues,
            backend="iverilog",
            backend_version=ctx.backend_version,
        )
