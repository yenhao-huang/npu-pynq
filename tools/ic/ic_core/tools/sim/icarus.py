"""Icarus Verilog simulation backend.

Kept as the four-state escape hatch. Verilator's two-state model silently
turns an X into a 0, which is exactly the class of reset and initialisation bug
worth catching; Icarus models X and Z properly at the cost of being far slower.

Icarus writes VCD, not FST, so runs from this backend hand the `debug` category
a `.vcd` handle. The debug contract reads both, which is the point of the
handle being a file rather than a format.
"""

from __future__ import annotations

from pathlib import Path

from ...process import run as run_process
from ...registry import backend
from . import SUMMARY_LINE_LIMIT, SimIn, SimOut
from . import probe as trace_probe
from .summary import summarize

WAVE_NAME = "wave.vcd"


@backend("sim", "icarus", requires="iverilog", version_cmd=["iverilog", "-V"])
class IcarusSim:
    def sim(self, params: SimIn, ctx) -> SimOut:
        work = ctx.run.work
        log = ctx.run.artifacts / "sim.log"
        binary = work / f"{params.tb}.vvp"
        sources = [str(Path(f)) for f in params.files]

        argv = ["iverilog", "-g2012", "-o", str(binary), "-s", params.tb]
        if params.trace:
            sources.append(str(trace_probe.write(
                work, params.tb, str(ctx.run.artifacts / WAVE_NAME), use_bind=False)))
            argv += ["-s", trace_probe.PROBE_MODULE]
        for directory in params.include_dirs:
            argv += ["-I" + directory]
        for define in params.defines:
            argv += ["-D" + define]
        argv += sources

        build = run_process(argv, log_path=log, cwd=ctx.cwd, timeout_s=params.timeout_s)
        if build.exit_code != 0 or not binary.exists():
            lines, truncated, _, fails = summarize(build.text(), SUMMARY_LINE_LIMIT)
            return SimOut(
                ok=False,
                built=False,
                exit_code=build.exit_code,
                timed_out=build.timed_out,
                duration_s=round(build.duration_s, 3),
                pass_count=0,
                fail_count=max(fails, 1),
                summary=lines,
                summary_truncated=truncated,
                wave=None,
                log=ctx.run.handle("sim.log"),
                backend="icarus",
                backend_version=ctx.backend_version,
            )

        run_argv = ["vvp", str(binary)] + [f"+{arg}" for arg in params.plusargs]
        sim = run_process(
            run_argv,
            log_path=log,
            cwd=ctx.cwd,
            timeout_s=params.timeout_s,
            append=True,
        )
        lines, truncated, passes, fails = summarize(sim.text(), SUMMARY_LINE_LIMIT)
        wave = ctx.run.artifacts / WAVE_NAME
        return SimOut(
            ok=sim.exit_code == 0 and fails == 0 and not sim.timed_out,
            built=True,
            exit_code=sim.exit_code,
            timed_out=sim.timed_out,
            duration_s=round(build.duration_s + sim.duration_s, 3),
            pass_count=passes,
            fail_count=fails,
            summary=lines,
            summary_truncated=truncated,
            wave=ctx.run.handle(WAVE_NAME) if wave.exists() else None,
            log=ctx.run.handle("sim.log"),
            backend="icarus",
            backend_version=ctx.backend_version,
        )
