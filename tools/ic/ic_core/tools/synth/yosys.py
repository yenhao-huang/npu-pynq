"""Yosys synthesis backend -- the `estimate` mode.

Runs `synth_xilinx` for the 7-series and reports the cell counts it prints.
The numbers are technology-mapped but not placed or routed, so they answer
"is this bigger than it was" and nothing more. `note` on the result says so,
because a LUT count that looks authoritative is worse than no number at all.
"""

from __future__ import annotations

import re

from ...process import run as run_process
from ...registry import backend
from . import SynthIn, SynthOut, Timing, Utilization

STAT_LINE = re.compile(r"^\s+(\w[\w$]*)\s+(\d+)\s*$")

#: Yosys reports mapped 7-series primitives; these map them onto the
#: vocabulary Vivado uses, so both modes fill the same `Utilization` fields.
#: INV is listed because on 7-series it is not free: it is placed as a LUT1.
#: Leaving it out reported an 8-bit counter (CARRY4 + 9 INV) as using no LUTs.
LUT_CELLS = ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6", "INV")
FF_CELLS = ("FDRE", "FDSE", "FDCE", "FDPE", "FDRE_1", "FDCE_1")
DSP_CELLS = ("DSP48E1", "DSP48E2", "DSP48A1")
BRAM_CELLS = ("RAMB18E1", "RAMB36E1", "RAMB18", "RAMB36")


def parse_stat(text: str) -> tuple[Utilization, list[str]]:
    """Read the final `=== design hierarchy ===` / `stat` block."""
    counts: dict[str, int] = {}
    total_cells = None
    memory_bits = None
    # `stat` prints one `=== <module> ===` block per module plus a hierarchy
    # roll-up. The last block carrying a cell count is the whole design;
    # earlier ones are submodules and would understate the result.
    blocks = re.split(r"^=== .* ===$", text, flags=re.M)
    tail = next(
        (b for b in reversed(blocks) if "Number of cells:" in b),
        text,
    )
    for raw in tail.splitlines():
        line = raw.rstrip()
        match = STAT_LINE.match(line)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + int(match.group(2))
            continue
        if "Number of cells:" in line:
            total_cells = int(line.split(":")[1].strip())
        elif "Estimated number of LCs:" in line:
            counts.setdefault("_lcs", int(line.split(":")[1].strip()))
        elif "Number of memory bits:" in line:
            memory_bits = int(line.split(":")[1].strip())

    parsed = total_cells is not None

    def total(names) -> int | None:
        # Zero, not null, once a stat block was read: "none of these cells"
        # is an answer. Null is kept for "the report could not be parsed".
        found = [counts[n] for n in names if n in counts]
        if found:
            return sum(found)
        return 0 if parsed else None

    utilization = Utilization(
        luts=total(LUT_CELLS),
        ffs=total(FF_CELLS),
        dsps=total(DSP_CELLS),
        brams=total(BRAM_CELLS),
        cells=total_cells,
        memory_bits=memory_bits,
    )
    summary = [
        line.strip()
        for line in tail.splitlines()
        if any(k in line for k in ("Number of cells:", "Number of wires:",
                                   "Estimated number of LCs:", "Number of memory bits:"))
    ]
    return utilization, summary[:8]


@backend("synth", "yosys", requires="yosys", version_cmd=["yosys", "-V"])
class YosysSynth:
    def synth(self, params: SynthIn, ctx) -> SynthOut:
        script = ctx.run.work / "synth.ys"
        lines = []
        for define in params.defines:
            name, _, value = define.partition("=")
            lines.append(f"verilog_defines -D{name}={value or 1}")
        include_flags = "".join(f" -I{d}" for d in params.include_dirs)
        for source in params.files:
            lines.append(f"read_verilog -sv{include_flags} {source}")
        lines += [
            f"hierarchy -check -top {params.top}",
            f"synth_xilinx -family xc7 -top {params.top}",
            "stat",
        ]
        script.write_text("\n".join(lines) + "\n")

        log = ctx.run.artifacts / "synth.log"
        result = run_process(
            ["yosys", "-s", str(script)],
            log_path=log,
            cwd=ctx.cwd,
            timeout_s=params.timeout_s,
        )
        text = result.text()
        utilization, summary = parse_stat(text)
        if result.exit_code != 0:
            summary = [ln.strip() for ln in text.splitlines()
                       if "ERROR" in ln or "Error" in ln][:8] or summary
        return SynthOut(
            ok=result.exit_code == 0,
            mode="estimate",
            top=params.top,
            part=None,
            utilization=utilization,
            timing=Timing(met=None),
            summary=summary,
            report=ctx.run.handle("synth.log"),
            exit_code=result.exit_code,
            duration_s=round(result.duration_s, 3),
            backend="yosys",
            backend_version=ctx.backend_version,
            note=(
                "Estimate only: technology-mapped but not placed or routed, and no "
                "timing analysis. Use mode='full' before trusting area or timing."
            ),
        )
