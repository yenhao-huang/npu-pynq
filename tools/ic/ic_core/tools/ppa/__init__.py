"""ppa -- the category contract.

`synth` answers "does this fit, and does it meet timing on the Zynq-7020".
`ppa` answers a different question: "is this RTL a good circuit". It maps the
design onto a standard-cell library and reports the three numbers that travel
together in hardware design:

| Letter | Field            | Where it comes from                     |
| ------ | ---------------- | --------------------------------------- |
| P      | `power`          | OpenSTA power analysis, in watts        |
| P      | `area`           | summed standard-cell area, in um^2      |
| A      | `timing.fmax_mhz`| derived from slack against the constraint|

Two reasons it is its own category rather than a `synth` backend. Its
vocabulary is standard cells (um^2, watts), not FPGA primitives (LUTs, BRAMs),
so it cannot fill `Utilization`. And its target is a cell library the caller
names, not the part in `DEFAULT_PART` -- the numbers are a technology-neutral
yardstick for comparing two implementations, not a prediction about the board.

Both modes are pre-route, and `note` on every result says which. Nothing here
replaces `synth --mode full` for a statement about the PYNQ-Z1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..common import Backendable


class Area(BaseModel):
    cell_area_um2: float | None = Field(
        default=None, description="Summed standard-cell area of the mapped netlist"
    )
    die_area_um2: float | None = Field(
        default=None, description="Die area; null unless the floorplan ran (mode='placed')"
    )
    utilization_pct: float | None = Field(
        default=None, description="Cell area as a percentage of core area; null without a floorplan"
    )
    cells: int | None = Field(default=None, description="Mapped cell instance count")
    sequential_cells: int | None = Field(default=None, description="Of which are registers")


class Power(BaseModel):
    """Watts, as OpenSTA reports them.

    Estimated from an assumed input switching activity, not from a simulation
    trace, so treat it as a comparison between two designs rather than a
    prediction of what a board will draw.
    """

    total_w: float | None = None
    internal_w: float | None = None
    switching_w: float | None = None
    leakage_w: float | None = None
    sequential_w: float | None = Field(default=None, description="Register power, of the total")
    combinational_w: float | None = Field(default=None, description="Logic power, of the total")
    clock_w: float | None = Field(default=None, description="Clock network power, of the total")


class PpaTiming(BaseModel):
    clock_period_ns: float | None = Field(
        default=None, description="The constraint the slack below is measured against"
    )
    wns_ns: float | None = Field(
        default=None, description="Worst negative slack; negative means the constraint failed"
    )
    tns_ns: float | None = None
    met: bool | None = Field(default=None, description="Null when no timing path was analysed")
    fmax_mhz: float | None = Field(
        default=None,
        description="1000 / (period - wns). Pre-route and therefore optimistic; use it to rank designs, not to promise a frequency.",
    )


class PpaIn(Backendable):
    files: list[str] = Field(description="Synthesizable RTL. Do not include testbenches.")
    top: str = Field(description="Top module name")
    liberty: list[str] = Field(
        default_factory=list,
        description="Liberty (.lib) timing libraries to map onto. Falls back to IC_PDK_LIBERTY, a colon-separated path list.",
    )
    mode: Literal["estimate", "placed"] = Field(
        default="estimate",
        description="estimate = map and analyse the netlist, seconds, no wire load. placed = also floorplan and place, minutes, parasitics from placement.",
    )
    clock_port: str | None = Field(
        default=None,
        description="Name of the clock port. Without it a virtual clock is created and register-to-register paths are not constrained.",
    )
    clock_period_ns: float = Field(
        default=10.0,
        description="Target clock period. fmax is derived from the slack against it, so pick something the design can plausibly meet.",
    )
    sdc: str | None = Field(
        default=None,
        description="SDC constraints file, used instead of the generated create_clock.",
    )
    input_activity: float = Field(
        default=0.1,
        description="Assumed switching activity on input ports, 0..1. Power scales with it, so keep it identical across designs you compare.",
    )
    tech_lef: str | None = Field(
        default=None,
        description="Technology LEF. Required for mode='placed'; recommended for 'estimate' too. Falls back to IC_PDK_TECH_LEF.",
    )
    lef: list[str] = Field(
        default_factory=list,
        description="Standard-cell LEF files. Required for mode='placed'. Falls back to IC_PDK_LEF, a colon-separated path list.",
    )
    site: str | None = Field(
        default=None,
        description="Placement site name from the tech LEF. Required for mode='placed'. Falls back to IC_PDK_SITE.",
    )
    utilization: float = Field(
        default=40.0, description="Target core utilization percentage for the floorplan (mode='placed')"
    )
    hor_layer: str | None = Field(
        default=None,
        description="Horizontal metal layer for pin placement. Required for mode='placed'. Falls back to IC_PDK_HOR_LAYER.",
    )
    ver_layer: str | None = Field(
        default=None,
        description="Vertical metal layer for pin placement. Required for mode='placed'. Falls back to IC_PDK_VER_LAYER.",
    )
    include_dirs: list[str] = Field(default_factory=list, description="Include search path")
    defines: list[str] = Field(
        default_factory=list, description="Preprocessor defines, NAME or NAME=VALUE"
    )
    timeout_s: float = Field(default=3600.0, description="Kill the flow after this many seconds")

    @model_validator(mode="after")
    def _activity_is_a_fraction(self):
        # The PDK fields are *not* checked here: they fall back to the
        # environment, which is resolved at run time by the backend. Checking
        # them now would reject a call the environment can satisfy.
        if not 0.0 <= self.input_activity <= 1.0:
            raise ValueError("input_activity is a fraction between 0 and 1")
        return self


class PpaOut(BaseModel):
    ok: bool
    mode: str
    top: str
    area: Area
    power: Power
    timing: PpaTiming
    liberty: list[str] = Field(
        default_factory=list, description="The libraries actually mapped onto"
    )
    summary: list[str] = Field(
        default_factory=list, description="The few report lines that explain the numbers"
    )
    report: str | None = Field(default=None, description="Full log handle 'run_id/ppa.log'")
    netlist: str | None = Field(
        default=None, description="Mapped gate-level netlist handle 'run_id/netlist.v'"
    )
    exit_code: int = 0
    duration_s: float = 0.0
    backend: str
    backend_version: str
    note: str | None = Field(default=None, description="What the numbers do and do not include")
    run_id: str | None = None


from ...registry import Category, Op, register_category  # noqa: E402

CATEGORY = register_category(
    Category(
        name="ppa",
        summary="Power, area and fmax against a standard-cell library, for comparing two implementations.",
        default_backend="openroad",
        # OpenROAD and a standard-cell PDK are an extra, not part of the
        # lint/sim loop, so a machine without them is still a healthy one.
        optional=True,
        ops=[
            Op(
                name="ppa",
                In=PpaIn,
                Out=PpaOut,
                summary=(
                    "Map RTL onto a standard-cell library and report power, area and "
                    "fmax. mode='estimate' takes seconds and ignores wire load; "
                    "mode='placed' floorplans and places first, for parasitics. Both "
                    "are pre-route, and neither says anything about the FPGA -- use "
                    "`synth` for the Zynq-7020."
                ),
                long_running=True,
            )
        ],
    )
)

from . import openroad  # noqa: E402,F401
