"""Data contract shared by every tpu_gen module.

One DesignState flows through the pipeline; each module takes it and returns
an updated copy. PPA numbers may only come from OpenROAD -- see PPA.source.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


class FlowError(RuntimeError):
    """Any step that cannot complete. The flow never degrades, it fails."""


@dataclass
class PPATarget:
    """Target metrics as they appear in the TPU-Gen prompt format."""

    area: float           # um^2
    wns: float            # ns
    total_power: float    # W

    def as_prompt_metrics(self) -> str:
        return (
            f"Area: {self.area:g}, WNS: {self.wns:g}, "
            f"Total Power: {self.total_power:.2e}"
        )


@dataclass
class PPA:
    """Measured metrics. Only OpenROAD is allowed to produce these."""

    area: float           # um^2
    wns: float            # ns
    total_power: float    # W
    source: str           # must be "openroad"
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source != "openroad":
            raise FlowError(
                f"PPA.source must be 'openroad', got {self.source!r}. "
                "This flow has no fallback backend."
            )

    def distance(self, target: PPATarget) -> dict[str, float]:
        """Relative error per metric against a target."""

        def rel(got: float, want: float) -> float:
            return abs(got - want) / abs(want) if want else float("inf")

        return {
            "area": rel(self.area, target.area),
            "wns": rel(self.wns, target.wns),
            "total_power": rel(self.total_power, target.total_power),
        }


@dataclass
class DesignState:
    """Everything one iteration knows about the design under construction."""

    user_prompt: str
    target: PPATarget | None = None
    formatted_prompt: str | None = None
    vh_text: str | None = None
    config: dict = field(default_factory=dict)
    filelist: list[Path] = field(default_factory=list)
    gds: Path | None = None
    ppa: PPA | None = None
    errors: list[str] = field(default_factory=list)
    iteration: int = 0
    workdir: Path | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["filelist"] = [str(p) for p in self.filelist]
        d["gds"] = str(self.gds) if self.gds else None
        d["workdir"] = str(self.workdir) if self.workdir else None
        return json.dumps(d, indent=2)

    def dump(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")
