"""synth -- the category contract.

Two backends that are deliberately *not* interchangeable. Yosys and Vivado
answer the same question at different precisions and costs, so the choice is
exposed as `mode` rather than hidden behind `backend`:

| mode       | backend | cost      | what it is for                               |
| ---------- | ------- | --------- | -------------------------------------------- |
| `estimate` | Yosys   | seconds   | a rough LUT/FF count inside the agent's loop |
| `full`     | Vivado  | minutes   | the only source of truth for timing and area |

An agent should call `estimate` freely and `full` rarely. A Yosys cell count is
useful for "did that rewrite double the area?"; it cannot tell you whether the
design meets timing on a Zynq-7020, because it never ran place and route.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..common import Backendable

#: PYNQ-Z1. Only `full` uses it; Yosys does not model a specific device.
DEFAULT_PART = "xc7z020clg400-1"


class Utilization(BaseModel):
    luts: int | None = None
    ffs: int | None = None
    dsps: int | None = None
    brams: int | None = None
    cells: int | None = Field(default=None, description="Total post-synthesis cell count")
    memory_bits: int | None = None


class Timing(BaseModel):
    wns_ns: float | None = Field(default=None, description="Worst negative slack; negative means timing failed")
    tns_ns: float | None = None
    met: bool | None = Field(default=None, description="Null when the backend does not do timing analysis")


class SynthIn(Backendable):
    files: list[str] = Field(description="Synthesizable RTL. Do not include testbenches.")
    top: str = Field(description="Top module name")
    mode: Literal["estimate", "full"] = Field(
        default="estimate",
        description="estimate = Yosys, seconds, rough area. full = Vivado, minutes, real timing and utilization.",
    )
    part: str = Field(default=DEFAULT_PART, description="Target device; only meaningful for mode='full'")
    include_dirs: list[str] = Field(default_factory=list, description="Include search path")
    defines: list[str] = Field(default_factory=list, description="Preprocessor defines, NAME or NAME=VALUE")
    timeout_s: float = Field(default=3600.0, description="Kill synthesis after this many seconds")

    @model_validator(mode="after")
    def _backend_follows_mode(self):
        # `mode` is the agent-facing knob; the backend is a consequence of it.
        # Setting `backend` explicitly still wins, so a test can force Yosys to
        # run in full mode or vice versa.
        if self.backend is None:
            self.backend = "yosys" if self.mode == "estimate" else "vivado"
        return self


class SynthOut(BaseModel):
    ok: bool
    mode: str
    top: str
    part: str | None = None
    utilization: Utilization
    timing: Timing
    summary: list[str] = Field(default_factory=list, description="The few report lines that explain the numbers")
    report: str | None = Field(default=None, description="Full report handle 'run_id/synth.log'")
    exit_code: int = 0
    duration_s: float = 0.0
    backend: str
    backend_version: str
    note: str | None = Field(
        default=None, description="Caveats, e.g. that an estimate did not run place and route"
    )
    run_id: str | None = None


from ...registry import Category, Op, register_category  # noqa: E402

CATEGORY = register_category(
    Category(
        name="synth",
        summary="Area and timing feedback, either a fast estimate or the authoritative run.",
        default_backend="yosys",
        ops=[
            Op(
                name="synth",
                In=SynthIn,
                Out=SynthOut,
                summary=(
                    "Synthesize RTL and report area and timing. mode='estimate' (Yosys) "
                    "returns rough LUT/FF counts in seconds and is safe to call in a loop; "
                    "mode='full' (Vivado) takes minutes and is the only trustworthy source "
                    "of timing for the target device."
                ),
                long_running=True,
            )
        ],
    )
)

from . import vivado, yosys  # noqa: E402,F401
