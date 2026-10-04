from __future__ import annotations

import re
from pathlib import Path

from prompt_formatter import AP_ADDERS, AP_MULTIPLIERS
from retrieval import collect_defines
from tpugen_types import FlowError

# Every .v in the library reads these; a header missing one does not elaborate.
REQUIRED_MACROS = (
    "ROUN_WIDTH", "NIBBLE_WIDTH", "DW", "WW", "M", "N",
    "MULT_DW", "ADDER_PARAM", "VBL",
)

SHARED_PRE_APPROX_MULTIPLIERS = frozenset((
    "DRUM_APTPU", "MITCHELL", "ROBA", "DRALM", "ASM",
    "ALM_MAA3", "ALM", "ALM_SOA", "ALM_LOA",
))

_IVERILOG = re.compile(r"^(?:.*[/\\])?(?P<file>[\w.]+):(?P<line>\d+):\s*error:\s*(?P<msg>.*)$")
_ORFS_ERROR = re.compile(r"^(?:\[ERROR\s+[\w-]+\]|Error:|ERROR:)\s*(?P<msg>.*)$")
_YOSYS_ERROR = re.compile(r"^ERROR:\s*(?P<msg>.*)$")


def summarize_elaboration(log: str, *, limit: int = 10) -> list[str]:
    """One line per distinct elaboration error, path noise stripped."""
    out: list[str] = []
    for line in log.splitlines():
        m = _IVERILOG.match(line.strip())
        if m:
            entry = f"{m['file']}:{m['line']}: {m['msg']}"
            if entry not in out:
                out.append(entry)
    return out[:limit]


def summarize_orfs_log(log_file: Path, *, limit: int = 10) -> list[str]:
    """Pull the errors out of an ORFS run log.

    Falls back to the last few lines when nothing matches, so a crash with an
    unusual message is still reported rather than silently dropped.
    """
    if not log_file.exists():
        return [f"OpenROAD log missing: {log_file}"]

    text = log_file.read_text(errors="replace")
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        for pattern in (_ORFS_ERROR, _YOSYS_ERROR):
            m = pattern.match(line)
            if m and m["msg"] and m["msg"] not in out:
                out.append(m["msg"])
                break
    if not out:
        out = [ln for ln in text.splitlines()[-5:] if ln.strip()]
    return out[:limit]


def check_macros(vh_text: str) -> list[str]:
    """Check a generated header before anything is staged.

    Reports what is wrong rather than patching in defaults: a header the model
    got wrong should come back as feedback, not be quietly repaired into a
    design nobody asked for.
    """
    try:
        defines = collect_defines(vh_text)
    except FlowError as exc:
        return [f"header does not preprocess: {exc}"]

    problems = [
        f"missing `define {name}" for name in REQUIRED_MACROS if name not in defines
    ]

    multipliers = [m for m in AP_MULTIPLIERS if m in defines]
    adders = [a for a in AP_ADDERS if a in defines]
    if not multipliers:
        problems.append("no multiplier selected; define exactly one of " + ", ".join(AP_MULTIPLIERS))
    if not adders and "ACCURATE_ACCUMULATE" not in defines:
        problems.append("no adder selected; define one of " + ", ".join(AP_ADDERS))

    if any(m in defines for m in SHARED_PRE_APPROX_MULTIPLIERS) and \
            "SHARED_PRE_APPROX" not in defines:
        problems.append(
            "selected multiplier requires `define SHARED_PRE_APPROX"
        )

    for name in ("DW", "WW", "M", "N", "MULT_DW"):
        raw = defines.get(name, "").split()[0] if defines.get(name, "").strip() else ""
        if raw and not raw.isdigit():
            problems.append(f"`define {name} must be a positive integer, got {raw!r}")

    def as_int(name: str) -> int | None:
        raw = defines.get(name, "").split()
        return int(raw[0]) if raw and raw[0].isdigit() else None

    dw, ww, mult_dw = as_int("DW"), as_int("WW"), as_int("MULT_DW")
    if dw and ww and mult_dw and mult_dw > min(dw, ww):
        problems.append(
            f"MULT_DW ({mult_dw}) exceeds min(DW, WW) ({min(dw, ww)})"
        )
    return problems
