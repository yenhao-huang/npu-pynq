"""lint -- the category contract.

The schema belongs to the category, not to any tool. `verilator.py` only has
to translate Verilator's diagnostics into `LintOut`; swapping in Verible later
changes that one file and nothing an agent sees.

Lint output is the one result returned in full. A design with 200 warnings
produces a few KB, and truncating it would hide the warning that mattered.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..common import Backendable

Severity = Literal["error", "warning", "info"]


class Issue(BaseModel):
    file: str = Field(description="Path to the source file, relative to the project root")
    line: int | None = Field(default=None, description="1-based line number")
    column: int | None = Field(default=None, description="1-based column number")
    severity: Severity = Field(description="error, warning, or info")
    code: str | None = Field(default=None, description="Backend rule code, e.g. WIDTH or UNUSEDSIGNAL")
    message: str = Field(description="One-line human-readable description")


class LintIn(Backendable):
    files: list[str] = Field(description="SystemVerilog/Verilog source files to lint")
    top: str | None = Field(default=None, description="Top module name; inferred by the backend when omitted")
    include_dirs: list[str] = Field(default_factory=list, description="Directories added to the include search path")
    defines: list[str] = Field(default_factory=list, description="Preprocessor defines, each as NAME or NAME=VALUE")
    strict: bool = Field(default=True, description="Enable the backend's full warning set (-Wall)")


class LintOut(BaseModel):
    ok: bool = Field(description="True when there are no errors (warnings do not fail a lint)")
    error_count: int
    warning_count: int
    issues: list[Issue] = Field(description="Every diagnostic, in source order")
    backend: str
    backend_version: str
    run_id: str | None = None


from ...registry import Category, Op, register_category  # noqa: E402

CATEGORY = register_category(
    Category(
        name="lint",
        summary="Static checks on RTL. Seconds, no elaboration of a testbench.",
        default_backend="verilator",
        ops=[
            Op(
                name="lint",
                In=LintIn,
                Out=LintOut,
                summary=(
                    "Lint RTL sources and return every diagnostic with file, line, "
                    "severity and rule code. Run this after every RTL edit; it is the "
                    "cheapest signal available."
                ),
            )
        ],
    )
)

from . import iverilog, verilator  # noqa: E402,F401
