"""Complete binary factorial contrasts for source-bound physical records."""
from itertools import combinations, product
from statistics import mean
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from ...errors import InvalidInput
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint, sources
from . import CATEGORY
from .exploration import Result
from .physical_analysis import ClockRecord, compatible


class AblationCell(BaseModel):
    model_config = ConfigDict(extra='forbid')
    levels: dict[str, Literal[0, 1]]
    configuration: dict[str, str | int | bool]
    status: Literal['measured', 'failed']
    failure: str | None = None
    files: list[str] = Field(min_length=1)
    top: str = Field(pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    records: list[ClockRecord] = Field(default_factory=list, max_length=20)


class AblationIn(ExplorationInput):
    backend: str | None = Field(default='ablation', description='Full binary factorial physical contrasts.')
    factors: dict[str, tuple[str | int | bool, str | int | bool]] = Field(description='One to four configuration keys, each with ordered low/high levels.')
    cells: list[AblationCell] = Field(min_length=2, max_length=16, description='All 2**N combinations, including failed cells; repeat indices form matched blocks.')


def summary(values):
    return dict(per_repeat=values, mean=mean(values), min=min(values), max=max(values))


@backend('synth', 'ablation')
class Ablation:
    def candidate_ablation(self, p, ctx):
        names=list(p.factors)
        if not 1 <= len(names) <= 4 or any(a == b for a,b in p.factors.values()):
            raise InvalidInput('Require one to four distinct binary configuration factors')
        cells={}
        common=None
        identifiers=set()
        anchor=None
        repeats=None
        failures=[]
        for cell in p.cells:
            if set(cell.levels) != set(names): raise InvalidInput('Every cell must specify every factor')
            key=tuple(cell.levels[n] for n in names)
            if key in cells: raise InvalidInput('Duplicate factorial cell')
            cells[key]=cell
            for name in names:
                if name not in cell.configuration or cell.configuration[name] != p.factors[name][cell.levels[name]]:
                    raise InvalidInput('Factor levels disagree with configuration')
            fixed={k:v for k,v in cell.configuration.items() if k not in names}
            if common is None: common=fixed
            if fixed != common: raise InvalidInput('Nonfactor configuration differs between cells')
            sha=fingerprint(sources(cell.files,ctx.cwd),cell.top)
            if cell.status == 'failed':
                if not cell.failure or cell.records: raise InvalidInput('Failed cells need a reason and no complete record set')
                failures.append(dict(levels=cell.levels,reason=cell.failure,source_sha256=sha))
                continue
            if cell.failure or len(cell.records) < 3: raise InvalidInput('Measured cells need at least three records and no failure')
            if repeats is None: repeats=len(cell.records)
            if len(cell.records) != repeats: raise InvalidInput('Factorial cells need equal repeat counts')
            first=cell.records[0]
            for record in cell.records:
                if record.source_sha256 != sha: raise InvalidInput('Physical record does not match current source bytes')
                if anchor is None: anchor=record
                compatible(anchor,record)
                if (record.latency_cycles,record.initiation_interval)!=(first.latency_cycles,first.initiation_interval):
                    raise InvalidInput('Cycle contract changes between repeats')
                identity=record.evidence['metrics']
                if identity in identifiers: raise InvalidInput('Duplicate physical execution across cells or repeats')
                identifiers.add(identity)
        expected=set(product((0,1),repeat=len(names)))
        if set(cells)!=expected: raise InvalidInput('Incomplete factorial design; declare failed cells explicitly')
        base=dict(factors=p.factors,fixed_configuration=common,repeats=repeats,
                  scope=None if anchor is None else anchor.stage,
                  complete=not failures,failures=failures,independent_statistical_samples=False)
        if failures:
            return Result(ok=False,data=base,note='Failed combinations are retained. No effects are estimated from an incomplete factorial design.')
        metrics=('luts','ffs','dsps','brams','throughput_mtransactions_s','latency_cycles','initiation_interval')
        effects={}
        for metric in metrics:
            def value(key,r):
                row=cells[key].records[r]
                return getattr(row,metric) if metric in ('latency_cycles','initiation_interval') else row.metrics[metric]
            conditional=[]
            for i,name in enumerate(names):
                others=[j for j in range(len(names)) if j!=i]
                for setting in product((0,1),repeat=len(others)):
                    low=[0]*len(names)
                    for j,v in zip(others,setting): low[j]=v
                    high=low.copy();high[i]=1
                    conditional.append(dict(factor=name,other_levels={names[j]:v for j,v in zip(others,setting)},
                                            **summary([value(tuple(high),r)-value(tuple(low),r) for r in range(repeats)])))
            contrasts=[]
            for order in range(1,len(names)+1):
                for subset in combinations(range(len(names)),order):
                    # Average finite difference over every setting of the other factors.
                    divisor=2**(len(names)-order)
                    values=[sum((-1)**(order-sum(key[i] for i in subset))*value(key,r) for key in sorted(cells))/divisor for r in range(repeats)]
                    contrasts.append(dict(factors=[names[i] for i in subset],order=order,**summary(values)))
            effects[metric]=dict(conditional_effects=conditional,contrasts=contrasts)
        base.update(effects=effects,sources=[dict(levels=c.levels,source_sha256=c.records[0].source_sha256,
                    physical_handles=[r.evidence['metrics'] for r in c.records]) for c in p.cells])
        return Result(data=base,note='Effects are high-minus-low in native metric units; pair interactions are averaged differences of differences. Negative resource effects save resources; positive throughput effects improve throughput. All interactions up to factor count are retained. Matched reruns measure reproducibility, not statistical significance. Current source hashes and physical conditions are checked; declared configuration labels, report authenticity and correctness require independent audit. This retrospective decomposition does not establish a new acceptance win or predeclared objective.')


CATEGORY.ops.append(Op('candidate_ablation',AblationIn,Result,'Decompose complete physical factorial studies into conditional, main and interaction effects.'))
