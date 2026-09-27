"""Multi-resource and repeated-run checks for clocked physical studies."""
import math
from statistics import mean
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ...errors import InvalidInput
from ...registry import Op, backend
from ..exploration_common import ExplorationInput
from . import CATEGORY
from .exploration import Result


class ClockRecord(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    name: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    tool: str = Field(min_length=1)
    version: str = Field(min_length=1)
    build: str = Field(min_length=1)
    part: str = Field(min_length=1)
    stage: Literal['routed_clocked_ooc','synthetic_fixture']
    period_ns: float = Field(gt=0)
    directive: str = Field(min_length=1)
    optimization_mode: str | None = None
    implementation_threads: int | None = Field(default=None,ge=1,le=8)
    latency_cycles: int = Field(ge=1)
    initiation_interval: int = Field(ge=1)
    metrics: dict[str,float]
    evidence: dict[str,str]

    @model_validator(mode='after')
    def physical_contract(self):
        required={'luts','ffs','dsps','brams','slack_ns','critical_period_ns','estimated_fmax_mhz','throughput_mtransactions_s'}
        if not required<=self.metrics.keys(): raise ValueError('missing required clocked metrics')
        if not {'metrics','utilization','timing','log'}<=self.evidence.keys() or any(not v for v in self.evidence.values()):
            raise ValueError('missing physical evidence handles')
        for name,value in self.metrics.items():
            if not math.isfinite(value) or (name!='slack_ns' and value<0):
                raise ValueError('nonfinite or negative physical metric')
        for name in ('luts','ffs','dsps'):
            if self.metrics[name]!=int(self.metrics[name]): raise ValueError('resource count must be integral')
        if self.metrics['brams']*2!=int(self.metrics['brams']*2): raise ValueError('BRAM tiles must use half-tile increments')
        period=self.metrics['critical_period_ns']
        if period<=0 or not math.isclose(period,self.period_ns-self.metrics['slack_ns'],rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError('critical period disagrees with clock and slack')
        expected=1000/period
        if not math.isclose(expected,self.metrics['estimated_fmax_mhz'],rel_tol=1e-6):
            raise ValueError('Fmax disagrees with critical period')
        if not math.isclose(expected/self.initiation_interval,self.metrics['throughput_mtransactions_s'],rel_tol=1e-6):
            raise ValueError('throughput disagrees with initiation interval')
        return self


class ClockPair(BaseModel):
    model_config=ConfigDict(extra='forbid')
    baseline: ClockRecord
    candidate: ClockRecord


class TradeoffIn(ExplorationInput):
    backend: str | None = Field(default='physical_analysis',description='Clocked resource-tradeoff analysis backend.')
    baseline: ClockRecord = Field(description='Baseline routed record with resource classes, II, latency and evidence.')
    candidate: ClockRecord = Field(description='Candidate routed record under matched physical conditions.')


class RepeatsIn(ExplorationInput):
    backend: str | None = Field(default='physical_analysis',description='Paired physical reproducibility analysis backend.')
    pairs: list[ClockPair] = Field(min_length=1,max_length=100,description='Matched independent executions; duplicate report handles are rejected.')
    objective: Literal['area','throughput'] = Field(description='Objective predeclared before seeing measurements.')


CONTEXT=('tool','version','build','part','stage','period_ns','directive','implementation_threads','optimization_mode')
RESOURCES=('luts','ffs','dsps','brams')


def compatible(a,b):
    differences=[field for field in CONTEXT if getattr(a,field)!=getattr(b,field)]
    if differences: raise InvalidInput('Incompatible physical conditions: '+', '.join(differences))


def compare(a,b):
    compatible(a,b)
    x,y=a.metrics,b.metrics
    throughput_ratio=y['throughput_mtransactions_s']/x['throughput_mtransactions_s']
    lut_ratio=y['luts']/x['luts'] if x['luts'] else None
    delta={r:y[r]-x[r] for r in RESOURCES}
    growth={r:(y[r]/x[r]-1)*100 if x[r] else (0 if y[r]==0 else None) for r in RESOURCES}
    changed=a.source_sha256!=b.source_sha256
    area_win=bool(changed and lut_ratio is not None and lut_ratio<=.85 and throughput_ratio>=.95 and y['dsps']<=x['dsps'] and y['brams']<=x['brams'])
    throughput_win=bool(changed and throughput_ratio>=1.15 and all(y[r]<=x[r]*1.25 for r in RESOURCES))
    reasons=[]
    if not changed: reasons.append('Identical source hashes cannot establish an optimization.')
    if lut_ratio is None: reasons.append('Zero LUT baseline has no percentage area reduction.')
    if throughput_ratio<.95: reasons.append('Throughput regression exceeds the 5% area-win limit.')
    if y['dsps']>x['dsps'] or y['brams']>x['brams']: reasons.append('DSP or BRAM growth prevents an unconditional LUT-area win.')
    if any(y[r]>x[r]*1.25 for r in RESOURCES): reasons.append('Resource growth exceeds the 25% throughput-win limit.')
    return dict(lut_reduction_pct=None if lut_ratio is None else (1-lut_ratio)*100,
                throughput_gain_pct=(throughput_ratio-1)*100,throughput_ratio=throughput_ratio,
                resource_delta=delta,resource_growth_pct=growth,
                latency_cycles=dict(baseline=a.latency_cycles,candidate=b.latency_cycles),
                initiation_interval=dict(baseline=a.initiation_interval,candidate=b.initiation_interval),
                area_gate=area_win,throughput_gate=throughput_win,reasons=reasons)


@backend('synth','physical_analysis')
class PhysicalAnalysis:
    def resource_tradeoff(self,p,ctx):
        data=compare(p.baseline,p.candidate)
        data['scope']=p.baseline.stage
        return Result(data=data,note='PPA-only gate at 15% gain with explicit resource and throughput limits. Correctness, source/report authenticity and family diversity must be audited separately.')

    def paired_repeat_summary(self,p,ctx):
        first=p.pairs[0]
        identifiers=[]
        comparisons=[]
        for pair in p.pairs:
            for role in ('baseline','candidate'):
                current=getattr(pair,role)
                anchor=getattr(first,role)
                compatible(anchor,current)
                if (current.source_sha256,current.latency_cycles,current.initiation_interval)!=(anchor.source_sha256,anchor.latency_cycles,anchor.initiation_interval):
                    raise InvalidInput('Repeated executions must preserve each design and cycle contract')
                identifiers.append(current.evidence['metrics'])
            comparisons.append(compare(pair.baseline,pair.candidate))
        if len(set(identifiers))!=len(identifiers): raise InvalidInput('Duplicate physical evidence cannot count as repeated execution')
        ratios=[]
        for pair in p.pairs:
            a,b=pair.baseline.metrics,pair.candidate.metrics
            if p.objective=='area':
                if a['luts']==0 or b['luts']==0: raise InvalidInput('Geometric area ratio requires positive LUT counts')
                ratios.append(a['luts']/b['luts'])
            else: ratios.append(b['throughput_mtransactions_s']/a['throughput_mtransactions_s'])
        summary={}
        for key in ('lut_reduction_pct','throughput_gain_pct'):
            values=[c[key] for c in comparisons]
            summary[key]=None if any(v is None for v in values) else dict(min=min(values),max=max(values),mean=mean(values))
        data=dict(repeats=len(p.pairs),objective=p.objective,comparisons=comparisons,summary=summary,
                  objective_geometric_benefit=math.exp(mean(math.log(r) for r in ratios)),
                  area_gate=len(p.pairs)>=3 and all(c['area_gate'] for c in comparisons),
                  throughput_gate=len(p.pairs)>=3 and all(c['throughput_gate'] for c in comparisons),
                  scope=first.baseline.stage,independent_statistical_samples=False)
        return Result(data=data,note='At least three complete pairs are required. Every repeat must meet the PPA gate; min/max report reproducibility, not statistical confidence. This does not establish whole-family or whole-project acceptance.')


CATEGORY.ops.append(Op('resource_tradeoff',TradeoffIn,Result,'Check clocked FPGA multi-resource tradeoffs and normalize throughput by II.'))
CATEGORY.ops.append(Op('paired_repeat_summary',RepeatsIn,Result,'Validate distinct source-matched paired runs and conservatively summarize reproducible PPA gains.'))
