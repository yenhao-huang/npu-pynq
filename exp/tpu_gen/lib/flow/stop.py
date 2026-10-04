from __future__ import annotations

from abc import ABC, abstractmethod

from tpugen_types import DesignState, PPATarget


class StopCondition(ABC):
    @abstractmethod
    def met(self, state: DesignState) -> bool: ...

    @property
    def description(self) -> str:
        return type(self).__name__


class MaxIterations(StopCondition):
    def __init__(self, n: int):
        self.n = n

    def met(self, state: DesignState) -> bool:
        return state.iteration + 1 >= self.n

    @property
    def description(self) -> str:
        return f"reached {self.n} iterations"


class NoErrors(StopCondition):
    """Stop as soon as one iteration completes with GDSII and no errors."""

    def met(self, state: DesignState) -> bool:
        return not state.errors and state.ppa is not None

    @property
    def description(self) -> str:
        return "clean iteration with GDSII"


class PPAWithinTarget(StopCondition):
    def __init__(self, target: PPATarget, tolerance: float = 0.1):
        self.target = target
        self.tolerance = tolerance

    def met(self, state: DesignState) -> bool:
        if state.ppa is None:
            return False
        return all(v <= self.tolerance for v in state.ppa.distance(self.target).values())

    @property
    def description(self) -> str:
        return f"PPA within {self.tolerance:.0%} of target"
