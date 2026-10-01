"""sim -- the category contract.

A simulation produces two things an agent must never receive in full: a
waveform of hundreds of megabytes and a log of tens of megabytes. Both stay on
disk. `SimOut` returns a verdict, the handful of log lines that explain it, and
two handles.

The handles are the hinge of the whole design. `wave` goes to the `debug`
category for programmatic queries and to `view` for a human; the agent itself
never looks at a waveform.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..common import Backendable

#: How many log lines may reach the agent. Chosen so a failing regression with
#: a dozen distinct failures still fits, while a log that prints per-cycle
#: cannot flood the context.
SUMMARY_LINE_LIMIT = 40


class SimIn(Backendable):
    files: list[str] = Field(
        description="All sources to compile: RTL and the testbench together"
    )
    tb: str = Field(description="Top-level testbench module name, e.g. tb_npu_pe")
    top: str | None = Field(
        default=None,
        description="Name of the design under test. Recorded for run history; optional.",
    )
    trace: bool = Field(
        default=True,
        description="Write a waveform. Leave on: it costs little and is the input to every debug query.",
    )
    include_dirs: list[str] = Field(default_factory=list, description="Include search path")
    defines: list[str] = Field(default_factory=list, description="Preprocessor defines, NAME or NAME=VALUE")
    timeout_s: float = Field(default=900.0, description="Kill the simulation after this many seconds")
    plusargs: list[str] = Field(
        default_factory=list, description="Runtime plusargs passed to the simulation, without the leading +"
    )


class SimOut(BaseModel):
    ok: bool = Field(description="True when the build succeeded, the run exited 0, and no failure line was seen")
    built: bool = Field(description="False when compilation failed; check `summary` for the compile errors")
    exit_code: int
    timed_out: bool
    duration_s: float
    pass_count: int = Field(description="Lines matching a PASS pattern")
    fail_count: int = Field(description="Lines matching a FAIL/ERROR/assertion pattern")
    summary: list[str] = Field(
        description=f"Up to {SUMMARY_LINE_LIMIT} log lines that explain the verdict, not the whole log"
    )
    summary_truncated: bool = False
    wave: str | None = Field(
        default=None,
        description="Waveform handle 'run_id/wave.fst'. Pass to signals/value_at/value_range/first_mismatch/show_wave.",
    )
    log: str | None = Field(default=None, description="Full log handle 'run_id/sim.log'")
    backend: str
    backend_version: str
    run_id: str | None = None


from ...registry import Category, Op, register_category  # noqa: E402

CATEGORY = register_category(
    Category(
        name="sim",
        summary="Elaborate and run a testbench, keeping the waveform and log on disk.",
        default_backend="verilator",
        ops=[
            Op(
                name="sim",
                In=SimIn,
                Out=SimOut,
                summary=(
                    "Compile and run a testbench. Returns a pass/fail verdict, the log "
                    "lines that explain it, and a waveform handle to query with "
                    "first_mismatch/value_at/value_range/signals."
                ),
                long_running=True,
            )
        ],
    )
)

from . import icarus, verilator  # noqa: E402,F401
