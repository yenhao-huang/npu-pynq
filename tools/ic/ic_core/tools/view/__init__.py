"""view -- the category contract.

The only tool here that is not for the agent. GTKWave is GUI-only: nothing can
be read back out of it programmatically, so an agent calling `show_wave` is
handing a question to a person, not answering one.

What makes that handoff worth automating is the `.gtkw` save file. Without one,
opening a trace means clicking through a signal tree to find the three signals
that matter and then hunting for the right time. With one, the window opens
already showing those signals, in that order, centred on the cycle in question.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..common import Backendable


class ShowWaveIn(Backendable):
    wave: str = Field(description="Waveform handle from a sim run, e.g. '166d08/wave.fst', or a path")
    signals: list[str] = Field(
        default_factory=list,
        description="Signals to preselect, in display order. All top-level signals when empty.",
    )
    center_cycle: int | None = Field(
        default=None, description="Cycle to centre the view on, typically the one first_mismatch reported"
    )
    center_time: int | None = Field(default=None, description="Raw time to centre on; overrides `center_cycle`")
    clock: str | None = Field(default=None, description="Clock used to convert `center_cycle` to a time")
    zoom_cycles: int = Field(default=40, ge=2, description="Roughly how many cycles the initial view spans")
    launch: bool = Field(
        default=True,
        description="Open the GUI. Set false to only generate the .gtkw save file, which is what CI and headless machines want.",
    )


class ShowWaveOut(BaseModel):
    savefile: str = Field(description="Path to the generated .gtkw save file")
    wave_path: str = Field(description="Resolved path of the waveform the save file refers to")
    command: str = Field(description="The exact command to open this view by hand")
    launched: bool = Field(description="False when the GUI was not started; `reason` says why")
    reason: str | None = Field(default=None, description="Why the GUI was not launched, e.g. no DISPLAY")
    signals: list[str] = Field(description="Signals written into the save file")
    center_time: int | None = None
    run_id: str | None = None


from ...registry import Category, Op, register_category  # noqa: E402

CATEGORY = register_category(
    Category(
        name="view",
        summary="Open a waveform for a person, preloaded with the right signals at the right time.",
        default_backend="gtkwave",
        ops=[
            Op(
                name="show_wave",
                In=ShowWaveIn,
                Out=ShowWaveOut,
                summary=(
                    "Generate a GTKWave save file and open the waveform for a human, "
                    "preselecting signals and centring on a cycle. Use this to hand a "
                    "problem to a person; it returns nothing you can reason about."
                ),
            )
        ],
    )
)

from . import gtkwave  # noqa: E402,F401
