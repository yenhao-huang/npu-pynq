"""debug -- the category contract.

Four ops, because there are four questions worth asking of a waveform without
looking at it:

| op               | question                                        |
| ---------------- | ----------------------------------------------- |
| `signals`        | what is in this trace?                          |
| `first_mismatch` | at which cycle did reference and DUT diverge?   |
| `value_at`       | what were these signals at that moment?         |
| `value_range`    | how did this signal get there?                  |

`first_mismatch` is the one that matters. Finding a divergence by eye means
opening a waveform; finding it this way turns a 184 MB trace into one line and
is the reason an agent never needs to see a picture.

Every op here is read-only and cheap, so none of them records a run. A
debugging session issues dozens; giving each its own run directory would bury
the simulation that produced the trace.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..common import Backendable

#: Caps on what may cross back to an agent. A waveform has millions of
#: transitions; these bound the answer, and `truncated` says when the bound bit.
MAX_SIGNALS = 500
MAX_POINTS = 200
MAX_CONTEXT = 8


class SignalInfo(BaseModel):
    path: str = Field(description="Full hierarchical path")
    width: int
    kind: str = Field(description="VCD variable type, e.g. logic, wire, integer")


class Value(BaseModel):
    bin: str | None = Field(default=None, description="Bit string, padded to the declared width")
    hex: str | None = None
    dec: float | int | None = Field(default=None, description="Unsigned value, null when any bit is X or Z")
    signed: float | int | None = Field(default=None, description="Two's-complement value")
    unknown: bool = Field(default=False, description="True when the value contains X or Z")
    raw: str | None = None


class WaveIn(Backendable):
    """Fields every waveform query shares."""

    wave: str = Field(description="Waveform handle from a sim run, e.g. '166d08/wave.fst', or a path")
    clock: str | None = Field(
        default=None,
        description="Clock signal used to convert times to cycles. Auto-detected from a 1-bit clk/clock signal when omitted.",
    )


class SignalsIn(WaveIn):
    scope: str | None = Field(default=None, description="Only signals under this hierarchical scope")
    pattern: str | None = Field(default=None, description="Case-insensitive substring or glob against the signal path")
    limit: int = Field(default=200, ge=1, le=MAX_SIGNALS, description="Maximum signals to return")


class SignalsOut(BaseModel):
    scopes: list[str] = Field(description="Hierarchical scopes present in the trace")
    signals: list[SignalInfo]
    total: int = Field(description="Signals matching before the limit was applied")
    truncated: bool
    timescale: str
    clock: str | None = Field(default=None, description="Signal used for cycle numbering")


class ValueAtIn(WaveIn):
    signals: list[str] = Field(description="Signal paths, or unique suffixes of them")
    cycle: int | None = Field(default=None, description="Clock cycle, counted from the first rising edge (0-based)")
    time: int | None = Field(default=None, description="Raw simulation time in timescale units; overrides `cycle`")


class ValueAtOut(BaseModel):
    time: int
    cycle: int | None
    values: dict[str, Value] = Field(description="Last value at or before the requested moment, keyed by the name you asked for")
    missing: list[str] = Field(default_factory=list, description="Names that matched no signal in the trace")


class ValueRangeIn(WaveIn):
    signal: str = Field(description="One signal path or unique suffix")
    from_cycle: int | None = Field(default=None, description="First cycle to report, inclusive")
    to_cycle: int | None = Field(default=None, description="Last cycle to report, inclusive")
    from_time: int | None = Field(default=None, description="First time to report; overrides `from_cycle`")
    to_time: int | None = Field(default=None, description="Last time to report; overrides `to_cycle`")
    max_points: int = Field(default=64, ge=1, le=MAX_POINTS, description="Cap on returned transitions")
    changes_only: bool = Field(default=True, description="Report transitions rather than one row per cycle")


class Point(BaseModel):
    time: int
    cycle: int | None
    value: Value


class ValueRangeOut(BaseModel):
    signal: str
    width: int
    points: list[Point]
    total: int = Field(description="Transitions in range before `max_points` was applied")
    truncated: bool


class FirstMismatchIn(WaveIn):
    ref: str = Field(description="Reference signal: the expected value, e.g. the golden model output")
    dut: str = Field(description="Signal under test, compared against `ref`")
    from_cycle: int = Field(default=0, description="Start comparing at this cycle")
    to_cycle: int | None = Field(default=None, description="Stop at this cycle; runs to the end when omitted")
    ignore_unknown: bool = Field(
        default=True,
        description="Treat a cycle where either side is X or Z as a match. Turn off to catch X-propagation.",
    )
    context: int = Field(default=4, ge=0, le=MAX_CONTEXT, description="Cycles of history to return around the divergence")


class FirstMismatchOut(BaseModel):
    found: bool = Field(description="False means ref and dut agreed across the whole range")
    time: int | None = None
    cycle: int | None = None
    ref: str = Field(description="Resolved reference signal path")
    dut: str = Field(description="Resolved DUT signal path")
    ref_value: Value | None = None
    dut_value: Value | None = None
    context: list[dict] = Field(
        default_factory=list,
        description="Cycles leading up to and including the divergence, each with both values",
    )
    compared_cycles: int = Field(default=0, description="How many cycles were actually examined")


from ...registry import Category, Op, register_category  # noqa: E402

CATEGORY = register_category(
    Category(
        name="debug",
        summary="Query a waveform programmatically. Turns gigabytes into a few lines.",
        default_backend="fst",
        ops=[
            Op(
                name="signals",
                In=SignalsIn,
                Out=SignalsOut,
                summary="List the scopes and signals in a waveform. Start here to learn the names the other debug ops take.",
                records_run=False,
            ),
            Op(
                name="first_mismatch",
                In=FirstMismatchIn,
                Out=FirstMismatchOut,
                summary=(
                    "Find the first cycle where a reference signal and a signal under test "
                    "disagree. This is the fastest way to localise a functional bug: one call "
                    "replaces scrolling a waveform."
                ),
                records_run=False,
            ),
            Op(
                name="value_at",
                In=ValueAtIn,
                Out=ValueAtOut,
                summary="Read several signals at one cycle or time. Use it to inspect state around a mismatch.",
                records_run=False,
            ),
            Op(
                name="value_range",
                In=ValueRangeIn,
                Out=ValueRangeOut,
                summary="Follow one signal's transitions over a cycle range, to see how it reached a bad value.",
                records_run=False,
            ),
        ],
    )
)

from . import fst, vcd  # noqa: E402,F401
