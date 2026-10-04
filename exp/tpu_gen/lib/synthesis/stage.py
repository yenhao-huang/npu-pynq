from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tpugen_types import FlowError

VH_NAME = "options_definitions.vh"  # the name every .v `include's


def stage_sources(filelist: list[Path], vh_text: str, dest: Path) -> list[Path]:
    """Copy the retrieved .v files and the generated header into dest."""
    if not filelist:
        raise FlowError("empty filelist; retrieval produced no sources")
    dest.mkdir(parents=True, exist_ok=True)
    staged = []
    for src in filelist:
        target = dest / src.name
        shutil.copyfile(src, target)
        staged.append(target)
    (dest / VH_NAME).write_text(vh_text, encoding="utf-8")
    return staged


@dataclass
class ElaborationResult:
    ok: bool
    log: str
    errors: list[str]


def elaborate(src_dir: Path, top: str, *, timeout: int = 600) -> ElaborationResult:
    """Elaborate the staged design with iverilog.

    Catches illegal macro combinations the LLM can emit (a multiplier whose
    module is absent, MULT_DW wider than DW) before an hour of place and
    route is spent on them.
    """
    if shutil.which("iverilog") is None:
        raise FlowError("iverilog not found; it is required for pre-flight checks")

    sources = sorted(src_dir.glob("*.v"))
    if not sources:
        raise FlowError(f"no .v files staged in {src_dir}")

    cmd = [
        "iverilog", "-g2005", "-I", str(src_dir), "-s", top,
        "-o", str(src_dir / "elab.out"), *[str(p) for p in sources],
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=src_dir
    )
    log = (proc.stdout + proc.stderr).strip()
    (src_dir / "elaborate.log").write_text(log or "(no output)\n", encoding="utf-8")
    (src_dir / "elab.out").unlink(missing_ok=True)

    errors = [
        line for line in log.splitlines()
        if ": error:" in line or line.startswith("error:")
    ]
    return ElaborationResult(ok=proc.returncode == 0, log=log, errors=errors)
