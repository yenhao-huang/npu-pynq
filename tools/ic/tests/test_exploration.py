"""Boundary and integration tests for paper-informed exploration."""
from copy import deepcopy
from pathlib import Path

import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.registry import all_op_names
from ic_core.tools.synth.exploration import Record

NEW = {'optimization_rules', 'width_advice', 'ppa_measure', 'ppa_provenance', 'ppa_compare',
       'ppa_pareto', 'ppa_reward', 'ppa_select', 'comb_check', 'rtl_evaluate'}
ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / 'exp/tool-exploration/fixtures'


def record(name='baseline', **metrics):
    return dict(name=name, source_sha256='a'*64, tool='fixture', version='1', part='test',
                constraint_ns=10, stage='synthetic_fixture', units={'luts':'count','delay_ns':'ns'},
                metrics=metrics or {'luts':100, 'delay_ns':10}, evidence='explicit synthetic test fixture')


def check_payload(candidate='candidate.sv'):
    return dict(reference_files=[str(FIX/'baseline.sv')], candidate_files=[str(FIX/candidate)], top='dut',
                inputs=[{'name':n,'width':4} for n in ('a','b','c')]+[{'name':'sel','width':1}],
                outputs=[{'name':'y','width':4}], combinational_contract=True)


def test_ten_new_ops():
    assert NEW <= set(all_op_names())


def test_compare_and_provenance(store):
    a, b = record(), record('candidate', luts=80, delay_ns=12)
    result=dispatch('ppa_compare', dict(baseline=a,candidate=b), store=store)
    assert result['data']['improvement_pct'] == {'luts':20,'delay_ns':-20}
    b['version']='2'
    assert not dispatch('ppa_provenance', dict(baseline=a,candidate=b), store=store)['ok']
    with pytest.raises(InvalidInput, match='incompatible'):
        dispatch('ppa_compare', dict(baseline=a,candidate=b), store=store)


@pytest.mark.parametrize('field,value', [('metrics',{'luts':float('nan'),'delay_ns':1}),
                                         ('metrics',{'luts':-1,'delay_ns':1}),
                                         ('metrics',{'luts':1.2,'delay_ns':1}),
                                         ('units',{'luts':'um2','delay_ns':'ns'})])
def test_bad_measurements_rejected(field,value):
    r=record(); r[field]=value
    with pytest.raises(ValueError): Record.model_validate(r)


def test_zero_baseline(store):
    out=dispatch('ppa_compare',dict(baseline=record(luts=0,delay_ns=1),candidate=record('b',luts=1,delay_ns=1)),store=store)
    assert out['data']['improvement_pct']['luts'] is None


def test_pareto_tradeoffs_and_ties(store):
    rows=[record('a'),record('b',luts=80,delay_ns=12),record('c',luts=120,delay_ns=15),record('d')]
    out=dispatch('ppa_pareto',dict(records=rows),store=store)
    assert out['data']=={'frontier':['a','b','d'],'dominated':['c']}
    with pytest.raises(InvalidInput): dispatch('ppa_pareto',dict(records=[rows[0],rows[0]]),store=store)


def test_reward_gates_and_missing_metric(store):
    p=dict(baseline=record(),candidate=record('b',luts=80,delay_ns=8),correctness_passed=True)
    assert dispatch('ppa_reward',p,store=store)['data']['reward']==pytest.approx(.2)
    assert dispatch('ppa_reward',dict(p,correctness_passed=False),store=store)['data']['reward'] is None
    assert not dispatch('ppa_reward',dict(p,limits={'delay_ns':7}),store=store)['data']['eligible']
    with pytest.raises(InvalidInput): dispatch('ppa_reward',dict(p,weights={'power_w':1}),store=store)
    with pytest.raises(InvalidInput): dispatch('ppa_reward',dict(p,weights={'luts':float('inf')}),store=store)


def test_budget_and_unvisited_selection(store):
    rows=[dict(name='a',mean_reward=.8,visits=10,estimated_cost_s=5),dict(name='b',mean_reward=0,visits=0,estimated_cost_s=8)]
    assert dispatch('ppa_select',dict(candidates=rows,budget_s=10),store=store)['data']['selected']=='b'
    assert dispatch('ppa_select',dict(candidates=rows,budget_s=6),store=store)['data']['selected']=='a'
    assert dispatch('ppa_select',dict(candidates=rows,budget_s=1),store=store)['data']['selected'] is None


@pytest.mark.parametrize('reward,cost,visits,exploration',[
    (1e308,1e-308,1,1),
    (1e308,1,1,1e308),
    (1,1,10**1000,1),
])
def test_selector_rejects_overflow_from_finite_inputs(store,reward,cost,visits,exploration):
    candidate=dict(name='overflow',mean_reward=reward,visits=visits,estimated_cost_s=cost)
    with pytest.raises(InvalidInput,match='finite numeric range'):
        dispatch('ppa_select',dict(candidates=[candidate],budget_s=1,exploration=exploration),store=store)


@pytest.mark.parametrize('bounds,operation,expected', [((0,15,0,15),'add',(0,30,5,False)),
    ((-8,7,-8,7),'mul',(-56,64,8,True)),((0,7,0,7),'sub',(-7,7,4,True)),
    ((-1,-1,0,0),'add',(-1,-1,1,True)),((0,0,0,0),'mul',(0,0,1,False))])
def test_width_boundaries(store,bounds,operation,expected):
    p=dict(zip(['a_min','a_max','b_min','b_max'],bounds),operation=operation,declared_width=16)
    d=dispatch('width_advice',p,store=store)['data']
    assert tuple(d[k] for k in ['min','max','required_width','signed'])==expected


def test_rule_sources_and_no_matches(store):
    rows=dispatch('optimization_rules',{'topics':['AREA']},store=store)['data']['rules']
    assert rows and all(r['preconditions'] and r['url'].startswith('https://') for r in rows)
    assert not dispatch('optimization_rules',{'topics':['unknown']},store=store)['data']['rules']


def test_real_comb_check_and_negative_control(store):
    import shutil
    if not shutil.which('iverilog'): pytest.skip('Icarus unavailable')
    good=dispatch('comb_check',check_payload(),store=store)
    assert good['ok'] and good['data']['vectors']==8192
    bad=dispatch('comb_check',check_payload('incorrect.sv'),store=store)
    assert not bad['ok'] and bad['data']['first_mismatch'] is not None


def test_checker_input_limits(store):
    p=check_payload(); p['inputs'][0]['width']=20
    with pytest.raises(InvalidInput): dispatch('comb_check',p,store=store)
    p=check_payload(); p['top']='dut;bad'
    with pytest.raises(InvalidInput): dispatch('comb_check',p,store=store)


def test_pipeline_stops_before_measurement(store, monkeypatch):
    import ic_core.tools.pipeline.rtlrewriter as pipeline
    calls=[]
    def fake(name,payload,**kwargs):
        calls.append(name)
        return {'ok':False,'run_id':'negative-control','data':{}}
    monkeypatch.setattr(pipeline,'dispatch',fake)
    out=dispatch('rtl_evaluate',check_payload(),store=store)
    assert not out['ok'] and calls==['comb_check']


def test_pipeline_binds_sources(store,monkeypatch):
    import ic_core.tools.pipeline.rtlrewriter as pipeline
    def fake(name,payload,**kwargs):
        if name=='comb_check':
            return {'ok':True,'run_id':'check','data':{'reference_sha256':'b'*64,'candidate_sha256':'c'*64}}
        return {'ok':True,'run_id':'measure','data':{'record':record()}}
    monkeypatch.setattr(pipeline,'dispatch',fake)
    out=dispatch('rtl_evaluate',check_payload(),store=store)
    assert not out['ok'] and out['data']['stage']=='source_changed'


def test_cli_accepts_complex_fields(store):
    import json
    import subprocess
    import sys
    rows=[dict(name='a',mean_reward=.2,visits=1,estimated_cost_s=5)]
    proc=subprocess.run([sys.executable,'-m','ic_cli.main','ppa_select','--candidates',json.dumps(rows[0]),'--budget-s','6'],
                        cwd=ROOT,capture_output=True,text=True,check=True)
    assert json.loads(proc.stdout)['data']['selected']=='a'
    proc=subprocess.run([sys.executable,'-m','ic_cli.main','ppa_compare','--baseline',json.dumps(record()),
                         '--candidate',json.dumps(record('b',luts=80,delay_ns=9))],cwd=ROOT,capture_output=True,text=True,check=True)
    assert json.loads(proc.stdout)['data']['improvement_pct']['luts']==20


def test_operation_defaults_route_to_implementations():
    from ic_core.registry import iter_ops
    for category,op in iter_ops():
        selected=op.In.model_fields['backend'].default or category.default_backend
        assert hasattr(category.backends[selected].impl(),op.name), f'{op.name}: {selected}'


def test_comb_checker_rejects_unknown_output(store,tmp_path):
    import shutil
    if not shutil.which('iverilog'): pytest.skip('Icarus unavailable')
    candidate=tmp_path/'unknown.sv'
    candidate.write_text("module dut(input [3:0] a,b,c,input sel,output [3:0] y); assign y=4'bxxxx; endmodule")
    payload=check_payload(); payload['candidate_files']=[str(candidate)]
    result=dispatch('comb_check',payload,store=store)
    assert not result['ok'] and result['data']['first_mismatch']==0


def test_measure_timeout_returns_no_record(store,monkeypatch):
    from types import SimpleNamespace
    import ic_core.tools.synth.measure as module
    from ic_core.registry import CATEGORIES
    entry=CATEGORIES['synth'].backends['ppa_vivado']
    monkeypatch.setattr(entry,'available',lambda:True)
    monkeypatch.setattr(entry,'version',lambda:'test')
    monkeypatch.setattr(module,'run_process',lambda *a,**kw:SimpleNamespace(exit_code=0,timed_out=True))
    result=dispatch('ppa_measure',dict(files=[str(FIX/'baseline.sv')],top='dut',name='test'),store=store)
    assert not result['ok'] and 'record' not in result['data']


def test_explicit_null_backend_uses_operation_default(store):
    out=dispatch('ppa_compare',dict(backend=None,baseline=record(),candidate=record('b',luts=80,delay_ns=9)),store=store)
    assert out['data']['improvement_pct']['luts']==20
    out=dispatch('optimization_rules',dict(backend=None,topics=['area']),store=store)
    assert out['data']['rules']
