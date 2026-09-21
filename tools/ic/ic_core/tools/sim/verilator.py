"""Verilator simulation backend -- the default.

`--binary --timing` matters. Verilator is cycle-based and two-state, but with
`--timing` it accepts the delay and event constructs the existing SystemVerilog
testbenches use, so they run unmodified. The two-state limit is real and is
documented in the category notes: Verilator will not reproduce an X-propagation
bug. When that matters, use the `icarus` backend, which is four-state.

The build happens in `work/`, the waveform and log land in `artifacts/`, and
only the verdict crosses back.
"""

from __future__ import annotations

from pathlib import Path

from ...errors import ToolFailed
from ...process import run as run_process
from ...registry import backend
from . import SUMMARY_LINE_LIMIT, SimIn, SimOut
from . import probe as trace_probe
from .summary import summarize

WAVE_NAME = "wave.fst"


@backend("sim", "verilator", requires="verilator", version_cmd=["verilator", "--version"])
class VerilatorSim:
    def sim(self, params: SimIn, ctx) -> SimOut:
        work = ctx.run.work
        log = ctx.run.artifacts / "sim.log"
        sources = [str(Path(f)) for f in params.files]

        argv = [
            "verilator",
            "--binary",
            "--timing",
            "-Wno-fatal",
            "--top-module",
            params.tb,
            "--Mdir",
            str(work),
        ]
        if params.trace:
            argv += ["--trace-fst", "--trace-structs"]
            sources.append(str(trace_probe.write(work, params.tb, WAVE_NAME)))
        for directory in params.include_dirs:
            argv += ["-I" + directory]
        for define in params.defines:
            argv += ["-D" + define]
        argv += sources

        build = run_process(argv, log_path=log, cwd=ctx.cwd, timeout_s=params.timeout_s)
        if build.exit_code != 0:
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
                backend="verilator",
                backend_version=ctx.backend_version,
            )

        binary = work / f"V{params.tb}"
        if not binary.exists():
            raise ToolFailed(
                f"verilator reported success but produced no {binary.name}",
                log=ctx.run.handle("sim.log"),
            )

        # The simulation runs inside artifacts/ so $dumpfile's relative path
        # puts the waveform exactly where the handle expects it.
        run_argv = [str(binary)] + [f"+{arg}" for arg in params.plusargs]
        sim = run_process(
            run_argv,
            log_path=log,
            cwd=ctx.run.artifacts,
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
            backend="verilator",
            backend_version=ctx.backend_version,
        )
