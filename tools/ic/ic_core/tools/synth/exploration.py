"""Paper-informed PPA analytics. Algorithms are adaptations, not paper replicas."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...registry import Op, backend
from ...errors import InvalidInput
from ..exploration_common import ExplorationInput
from . import CATEGORY

Metric = Literal['luts', 'ffs', 'delay_ns', 'power_w']


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    name: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    tool: str = Field(min_length=1)
    version: str = Field(min_length=1)
    part: str = Field(min_length=1)
    constraint_ns: float = Field(gt=0)
    stage: Literal['routed_combinational', 'synthetic_fixture']
    units: dict[Metric, str]
    metrics: dict[Metric, float]
    evidence: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_metrics(self):
        expected = {'luts': 'count', 'ffs': 'count', 'delay_ns': 'ns', 'power_w': 'W'}
        if not self.metrics or set(self.units) != set(self.metrics):
            raise ValueError('metrics and units must have the same nonempty keys')
        for key, value in self.metrics.items():
            if not math.isfinite(value) or value < 0 or self.units[key] != expected[key]:
                raise ValueError(f'invalid value or unit for {key}')
            if key in ('luts', 'ffs') and value != int(value):
                raise ValueError('resource counts must be integers')
        return self


class Result(BaseModel):
    ok: bool = True
    data: dict
    note: str
    run_id: str | None = None


class PairIn(ExplorationInput):
    backend: str | None = Field(default='exploration', description='PPA analytics backend.')
    baseline: Record = Field(description='Baseline measurement with provenance and units.')
    candidate: Record = Field(description='Candidate measurement with provenance and units.')


class FrontierIn(ExplorationInput):
    backend: str | None = Field(default='exploration', description='PPA analytics backend.')
    records: list[Record] = Field(min_length=1, max_length=1000, description='Compatible candidate measurements.')
    objectives: list[Metric] = Field(default=['luts', 'delay_ns'], min_length=1, description='Metrics to minimize.')

    @model_validator(mode='after')
    def unique_names(self):
        if len({r.name for r in self.records}) != len(self.records):
            raise ValueError('candidate names must be unique')
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError('objectives must be unique')
        return self


class RewardIn(PairIn):
    weights: dict[Metric, float] = Field(default={'luts': 0.5, 'delay_ns': 0.5}, description='Nonnegative normalized-improvement weights.')
    limits: dict[Metric, float] = Field(default_factory=dict, description='Hard upper bounds in the declared metric units.')
    correctness_passed: bool = Field(description='Whether the candidate passed the separately recorded correctness gate.')

    @model_validator(mode='after')
    def finite_weights(self):
        values = list(self.weights.values()) + list(self.limits.values())
        if not self.weights or any(not math.isfinite(v) or v < 0 for v in values) or not math.isfinite(sum(self.weights.values())) or sum(self.weights.values()) <= 0:
            raise ValueError('weights and limits must be finite and nonnegative, with positive total weight')
        return self


class Candidate(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    name: str = Field(min_length=1)
    mean_reward: float
    visits: int = Field(ge=0)
    estimated_cost_s: float = Field(gt=0)


class SelectIn(ExplorationInput):
    backend: str | None = Field(default='exploration', description='PPA analytics backend.')
    candidates: list[Candidate] = Field(min_length=1, description='Rewrite actions with measured reward history and cost estimates.')
    budget_s: float = Field(gt=0, allow_inf_nan=False, description='Remaining wall-clock budget in seconds.')
    exploration: float = Field(default=1.0, ge=0, allow_inf_nan=False, description='UCB exploration coefficient.')

    @model_validator(mode='after')
    def unique_names(self):
        if len({c.name for c in self.candidates}) != len(self.candidates):
            raise ValueError('candidate names must be unique')
        return self


def incompatibilities(a: Record, b: Record) -> list[str]:
    keys = ['tool', 'version', 'part', 'constraint_ns', 'stage', 'units']
    return [key for key in keys if getattr(a, key) != getattr(b, key)]


def comparable(a: Record, b: Record):
    differences = incompatibilities(a, b)
    if differences:
        raise InvalidInput('incompatible measurement context: ' + ', '.join(differences))


def improvements(a: Record, b: Record) -> dict:
    comparable(a, b)
    values = {key: None if value == 0 else 100 * ((value - b.metrics[key]) / value)
              for key, value in a.metrics.items()}
    if any(value is not None and not math.isfinite(value) for value in values.values()):
        raise InvalidInput('metric ratios exceed finite numeric range')
    return values


@backend('synth', 'exploration')
class Exploration:
    def ppa_provenance(self, params: PairIn, ctx):
        differences = incompatibilities(params.baseline, params.candidate)
        return Result(ok=not differences, data={'compatible': not differences, 'differences': differences},
                      note='Checks declared context; does not authenticate external reports or prove equivalence.')

    def ppa_compare(self, params: PairIn, ctx):
        return Result(data={'improvement_pct': improvements(params.baseline, params.candidate),
                            'delta': {k: params.candidate.metrics[k] - v for k, v in params.baseline.metrics.items()}},
                      note='Positive improvement means reduction. A zero baseline has no defined percentage.')

    def ppa_pareto(self, params: FrontierIn, ctx):
        for record in params.records:
            comparable(params.records[0], record)
            if not set(params.objectives) <= record.metrics.keys():
                raise InvalidInput('every objective must be measured')
        def dominates(a, b):
            return (all(a.metrics[k] <= b.metrics[k] for k in params.objectives)
                    and any(a.metrics[k] < b.metrics[k] for k in params.objectives))
        frontier = [r.name for r in params.records if not any(dominates(o, r) for o in params.records)]
        return Result(data={'frontier': frontier, 'dominated': [r.name for r in params.records if r.name not in frontier]},
                      note='All objectives are minimized; equal points are retained. No scalar score replaces the frontier.')

    def ppa_reward(self, params: RewardIn, ctx):
        delta = improvements(params.baseline, params.candidate)
        keys = set(params.weights) | set(params.limits)
        if not keys <= delta.keys():
            raise InvalidInput('weighted or constrained metric is missing')
        active = {k: v for k, v in params.weights.items() if v > 0}
        if any(delta[k] is None for k in active):
            raise InvalidInput('a weighted baseline metric is zero')
        violated = [k for k, limit in params.limits.items() if params.candidate.metrics[k] > limit]
        eligible = params.correctness_passed and not violated
        reward = sum((active[k] / sum(active.values())) * (delta[k] / 100) for k in active) if eligible else None
        return Result(data={'eligible': eligible, 'reward': reward, 'violated_limits': violated},
                      note='PPA-RTL-inspired weighted improvement; correctness and hard limits gate eligibility. Not its training loss.')

    def ppa_select(self, params: SelectIn, ctx):
        feasible = [c for c in params.candidates if c.estimated_cost_s <= params.budget_s]
        total = sum(c.visits for c in params.candidates)
        # Unvisited feasible actions are explored first, deterministically by cost/name.
        unseen = sorted((c for c in feasible if c.visits == 0), key=lambda c: (c.estimated_cost_s, c.name))
        scores = {c.name: (c.mean_reward + params.exploration * math.sqrt(2 * math.log(max(2, total)) / c.visits)) / c.estimated_cost_s
                  for c in feasible if c.visits}
        chosen = unseen[0].name if unseen else (min(scores, key=lambda n: (-scores[n], n)) if scores else None)
        return Result(data={'selected': chosen, 'scores': scores, 'unvisited': [c.name for c in unseen]},
                      note='Cost-normalized UCB adaptation inspired by RTLRewriter; estimates do not reserve or enforce execution time.')


for name, model, summary in [
    ('ppa_provenance', PairIn, 'Check that two PPA measurements use comparable conditions.'),
    ('ppa_compare', PairIn, 'Compute resource, delay and available power improvements with unit checks.'),
    ('ppa_pareto', FrontierIn, 'Retain nondominated PPA candidates for multiple minimization objectives.'),
    ('ppa_reward', RewardIn, 'Score correctness-gated PPA improvements with configurable constraints.'),
    ('ppa_select', SelectIn, 'Select a rewrite action under a cost budget using cost-normalized UCB.'),
]:
    CATEGORY.ops.append(Op(name, model, Result, summary))
