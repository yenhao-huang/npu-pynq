"""Adversarial validation of physical gates using explicit synthetic records."""
from copy import deepcopy
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput


def record(name,run,luts=100,ffs=100,dsps=0,brams=0,fmax=200,ii=1):
    return dict(name=name,source_sha256=('a' if name=='baseline' else 'b')*64,tool='fixture',version='1',build='fixture1',part='test',stage='synthetic_fixture',period_ns=5,directive='Default',latency_cycles=1,initiation_interval=ii,
                metrics=dict(luts=luts,ffs=ffs,dsps=dsps,brams=brams,slack_ns=5-1000/fmax,critical_period_ns=1000/fmax,estimated_fmax_mhz=fmax,throughput_mtransactions_s=fmax/ii),
                evidence={k:f'{run}/{k}' for k in ('metrics','utilization','timing','log')})


def pairs(count=3):
    return [dict(baseline=record('baseline',f'a{i}'),candidate=record('candidate',f'b{i}',luts=80,fmax=240)) for i in range(count)]


def test_area_and_throughput_with_resource_limits(store):
    p=pairs()[0]
    result=dispatch('resource_tradeoff',p,store=store)['data']
    assert result['area_gate'] and result['throughput_gate']
    p['candidate']=record('candidate','b',luts=80,dsps=1,fmax=240)
    result=dispatch('resource_tradeoff',p,store=store)['data']
    assert not result['area_gate'] and not result['throughput_gate']
    assert result['resource_growth_pct']['dsps'] is None


def test_ii_prevents_fake_speedup(store):
    p=dict(baseline=record('baseline','a'),candidate=record('candidate','b',luts=80,fmax=300,ii=2))
    out=dispatch('resource_tradeoff',p,store=store)['data']
    assert out['throughput_gain_pct']==-25
    assert not out['area_gate'] and not out['throughput_gate']


def test_zero_lut_candidate_reports_only_conservative_finite_bound(store):
    rows=pairs()
    for row in rows: row['candidate']['metrics']['luts']=0
    data=dispatch('paired_repeat_summary',dict(pairs=rows,objective='area'),store=store)['data']
    assert data['area_gate'] and data['zero_lut_candidate']
    assert data['objective_geometric_benefit'] is None
    assert data['objective_geometric_benefit_lower_bound']==pytest.approx(100)
    rows[0]['baseline']['metrics']['luts']=0
    with pytest.raises(InvalidInput,match='positive LUT baseline'):
        dispatch('paired_repeat_summary',dict(pairs=rows,objective='area'),store=store)


def test_three_pairs_required_and_each_repeat_must_win(store):
    two=dispatch('paired_repeat_summary',dict(pairs=pairs(2),objective='area'),store=store)['data']
    assert not two['area_gate']
    three=dispatch('paired_repeat_summary',dict(pairs=pairs(),objective='area'),store=store)['data']
    assert three['area_gate'] and three['throughput_gate']
    assert three['objective_geometric_benefit']==pytest.approx(1.25)
    rows=pairs(); rows[-1]['candidate']=record('candidate','b2',luts=99,fmax=201)
    failed=dispatch('paired_repeat_summary',dict(pairs=rows,objective='area'),store=store)['data']
    assert not failed['area_gate'] and not failed['throughput_gate']


@pytest.mark.parametrize('mutation',['duplicate','source','build','period','ii'])
def test_repeat_provenance_rejected(store,mutation):
    rows=pairs()
    if mutation=='duplicate': rows[1]=deepcopy(rows[0])
    elif mutation=='source': rows[1]['candidate']['source_sha256']='c'*64
    elif mutation=='build': rows[1]['baseline']['build']='different'
    elif mutation=='period': rows[1]['baseline']['period_ns']=6
    elif mutation=='ii': rows[1]['candidate']['initiation_interval']=2
    with pytest.raises(InvalidInput): dispatch('paired_repeat_summary',dict(pairs=rows,objective='area'),store=store)


@pytest.mark.parametrize('metric,value',[('estimated_fmax_mhz',999),('throughput_mtransactions_s',999),('luts',float('nan')),('brams',.3),('ffs',-1)])
def test_invalid_metrics_cannot_enter_gate(store,metric,value):
    p=pairs()[0];p['candidate']['metrics'][metric]=value
    with pytest.raises(InvalidInput): dispatch('resource_tradeoff',p,store=store)


def test_same_source_not_optimization(store):
    p=pairs()[0];p['candidate']['source_sha256']=p['baseline']['source_sha256']
    out=dispatch('resource_tradeoff',p,store=store)['data']
    assert not out['area_gate'] and not out['throughput_gate']


def test_fixed_and_minimum_latency_are_not_conflated(store):
    p=pairs()[0];p['candidate']['latency_kind']='minimum'
    with pytest.raises(InvalidInput,match='latency_kind'):
        dispatch('resource_tradeoff',p,store=store)
