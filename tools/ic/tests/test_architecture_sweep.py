"""Pipeline gating and checkpoint integrity; synthetic measurements are explicit."""
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.exploration_common import fingerprint
from ic_core.tools.pipeline.rtlrewriter import sweep


def payload():
    return dict(study_name='test_study',cases=[dict(name='w64',generator='synth_priority_encoder',baseline=dict(width=64,architecture='linear'),candidate=dict(width=64,architecture='tree'),objective='throughput')],verify_only=True)


def mock_execution(monkeypatch,proof=True,bad_measurement=False):
    calls=[]
    real=sweep.dispatch
    monkeypatch.setattr(sweep,'identity',lambda p,ctx:{'fixture_version':'1'})
    def invoke(op,p,**kwargs):
        calls.append(op)
        if op=='yosys_equivalence':
            return dict(ok=proof,run_id='synthetic-proof',data=dict(reference_sha256=fingerprint([Path(f) for f in p['reference_files']],p['top']),candidate_sha256=fingerprint([Path(f) for f in p['candidate_files']],p['top'])))
        if op=='clocked_ppa':
            sha='bad' if bad_measurement else fingerprint([Path(f) for f in p['files']],p['top'])
            return dict(ok=True,run_id='synthetic-measurement',data={'record':{'source_sha256':sha,'stage':'synthetic_fixture'}})
        return real(op,p,**kwargs)
    monkeypatch.setattr(sweep,'dispatch',invoke)
    return calls


def test_resume_reuses_verified_inputs(store,tmp_path,monkeypatch):
    calls=mock_execution(monkeypatch)
    first=dispatch('architecture_sweep',payload(),store=store,cwd=tmp_path)
    assert first['ok']
    count=len(calls)
    second=dispatch('architecture_sweep',payload(),store=store,cwd=tmp_path)
    assert second['ok'] and len(calls)==count
    assert first['data']['checkpoint_key']==second['data']['checkpoint_key']


def test_unproved_candidate_never_measured(store,tmp_path,monkeypatch):
    calls=mock_execution(monkeypatch,proof=False)
    p=payload(); p['verify_only']=False
    result=dispatch('architecture_sweep',p,store=store,cwd=tmp_path)
    assert not result['ok'] and 'clocked_ppa' not in calls
    assert result['data']['cases'][0]['status']=='correctness_not_established'


def test_altered_wrapper_invalidates_checkpoint(store,tmp_path,monkeypatch):
    mock_execution(monkeypatch)
    first=dispatch('architecture_sweep',payload(),store=store,cwd=tmp_path)
    import json
    checkpoint=Path(first['data']['checkpoint_directory'])/'w64-baseline-generate.json'
    generated=json.loads(checkpoint.read_text())['result']['data']
    with Path(generated['wrapper_path']).open('a') as stream: stream.write('// mutated\n')
    with pytest.raises(InvalidInput,match='changed since generation'):
        dispatch('architecture_sweep',payload(),store=store,cwd=tmp_path)


def test_measurement_source_binding(store,tmp_path,monkeypatch):
    mock_execution(monkeypatch,bad_measurement=True)
    p=payload(); p['verify_only']=False
    with pytest.raises(InvalidInput,match='Source changed'):
        dispatch('architecture_sweep',p,store=store,cwd=tmp_path)


def test_environment_changes_checkpoint_key():
    p=sweep.SweepIn.model_validate(payload())
    assert sweep.checkpoint_key(p,{'version':'1'}) != sweep.checkpoint_key(p,{'version':'2'})


def test_lock_rejects_concurrent_writer(tmp_path):
    with sweep.study_lock(tmp_path/'lock'):
        with pytest.raises(InvalidInput,match='already running'):
            with sweep.study_lock(tmp_path/'lock'):
                pytest.fail('second lock acquired')
    with sweep.study_lock(tmp_path/'lock'):
        pass

@pytest.mark.parametrize('exit_code,version,accepted',[(1,'vivado v2026.1\nSW Build 6511674',True),(1,'incomplete output',False),(2,'vivado v2026.1\nSW Build 6511674',False)])
def test_vivado_version_launcher_exit(monkeypatch,tmp_path,exit_code,version,accepted):
    from types import SimpleNamespace
    def run(argv,**kwargs):
        if '-version' in argv:
            return SimpleNamespace(exit_code=exit_code,timed_out=False,text=lambda:version)
        name='Yosys 0.23' if argv[0]=='yosys' else 'Icarus Verilog version 13.0'
        return SimpleNamespace(exit_code=0,timed_out=False,text=lambda:name)
    monkeypatch.setattr(sweep,'run_process',run)
    params=sweep.SweepIn.model_validate(dict(payload(),verify_only=False))
    ctx=SimpleNamespace(cwd=tmp_path,run=SimpleNamespace(artifacts=tmp_path))
    if accepted:
        assert sweep.identity(params,ctx)['vivado']['version'][0]=='vivado v2026.1'
    else:
        with pytest.raises(InvalidInput): sweep.identity(params,ctx)


def test_retry_is_bounded_and_preserves_failure(store,tmp_path,monkeypatch):
    calls=mock_execution(monkeypatch)
    original=sweep.dispatch
    attempts=0
    def fail_once(op,params,**kwargs):
        nonlocal attempts
        if op=='clocked_ppa':
            attempts+=1
            if attempts==1: return dict(ok=False,run_id='failed-attempt',data={'exit_code':1})
        return original(op,params,**kwargs)
    monkeypatch.setattr(sweep,'dispatch',fail_once)
    p=payload();p.update(verify_only=False,repeats=1,measurement_attempts=2)
    result=dispatch('architecture_sweep',p,store=store,cwd=tmp_path)
    assert result['ok'] and attempts==3
    row=result['data']['cases'][0]
    assert [r['ok'] for r in row['attempts']]==[False,True,True]
    assert len(row['records'][0])==2


def test_fifo_pipeline_checks_core_and_fixture(store,tmp_path,monkeypatch):
    mock_execution(monkeypatch)
    p=dict(study_name='fifo_checks',cases=[dict(name='d64',generator='synth_fifo',baseline=dict(width=16,depth=64,architecture='shift'),candidate=dict(width=16,depth=64,architecture='circular'),objective='area')],verify_only=True)
    result=dispatch('architecture_sweep',p,store=store,cwd=tmp_path)
    assert result['ok']
    assert len(result['data']['cases'][0]['checks'])==4
    assert result['data']['cases'][0]['protocol_contract']=='ready_valid_fifo'


def test_failed_measurements_stop_at_attempt_limit(store,tmp_path,monkeypatch):
    mock_execution(monkeypatch)
    original=sweep.dispatch
    attempts=0
    def fail(op,params,**kwargs):
        nonlocal attempts
        if op=='clocked_ppa':
            attempts+=1
            return dict(ok=False,run_id=f'failure-{attempts}',data={'exit_code':1})
        return original(op,params,**kwargs)
    monkeypatch.setattr(sweep,'dispatch',fail)
    p=payload();p.update(verify_only=False,repeats=3,measurement_attempts=2)
    result=dispatch('architecture_sweep',p,store=store,cwd=tmp_path)
    assert not result['ok'] and attempts==2
    row=result['data']['cases'][0]
    assert row['status']=='measurement_failed' and len(row['attempts'])==2

@pytest.mark.parametrize('sat_timeout,accepted',[(True,True),(False,False)])
def test_affine_sweep_requires_proof_and_retains_sat_result(store,tmp_path,monkeypatch,sat_timeout,accepted):
    monkeypatch.setattr(sweep,'identity',lambda p,ctx:{'fixture_version':'affine'})
    real=sweep.dispatch
    def invoke(op,p,**kwargs):
        if op in ('gf2_equivalence','yosys_equivalence'):
            return dict(ok=op=='gf2_equivalence',run_id='synthetic-'+op,data=dict(timed_out=sat_timeout if op=='yosys_equivalence' else False,reference_sha256=fingerprint([Path(f) for f in p['reference_files']],p['top']),candidate_sha256=fingerprint([Path(f) for f in p['candidate_files']],p['top'])))
        return real(op,p,**kwargs)
    monkeypatch.setattr(sweep,'dispatch',invoke)
    p=dict(study_name='affine',proof_engine='affine',verify_only=True,cases=[dict(name='crc',generator='synth_crc_parallel',baseline=dict(data_width=8,architecture='unrolled'),candidate=dict(data_width=8,architecture='shared'),objective='area')])
    result=dispatch('architecture_sweep',p,store=store,cwd=tmp_path)
    assert result['ok'] is accepted
    assert not result['data']['cases'][0]['sat_crosscheck']['ok']
    assert result['data']['cases'][0]['formal']['ok']


def test_affine_sweep_rejects_nonlinear_generator(store,tmp_path):
    with pytest.raises(InvalidInput,match='limited to CRC'):
        dispatch('architecture_sweep',dict(payload(),proof_engine='affine'),store=store,cwd=tmp_path)


@pytest.mark.parametrize('generator', ['synth_serial_multiplier','synth_divider'])
def test_iterative_pipeline_allows_verified_cycle_differences(store,tmp_path,monkeypatch,generator):
    calls=mock_execution(monkeypatch)
    p=dict(study_name='arithmetic_checks',cases=[dict(name='w16',generator=generator,baseline=dict(width=16,architecture='serial'),candidate=dict(width=16,architecture='radix4'),objective='throughput')],verify_only=False,repeats=1)
    result=dispatch('architecture_sweep',p,store=store,cwd=tmp_path)
    assert result['ok'],result
    row=result['data']['cases'][0]
    assert [r['data']['verified_initiation_interval'] for r in row['checks']]==[17,17,9,9]
    assert row['protocol_contract']=='ready_valid_arithmetic'
    assert calls.count('clocked_ppa')==2


def test_iterative_pipeline_rejects_failed_cycle_evidence(store,tmp_path,monkeypatch):
    calls=mock_execution(monkeypatch)
    real=sweep.dispatch
    def invoke(op,p,**kwargs):
        if op=='latency_throughput': return dict(ok=False,run_id='synthetic-bad-cycle',data={})
        return real(op,p,**kwargs)
    monkeypatch.setattr(sweep,'dispatch',invoke)
    p=dict(study_name='bad_cycles',cases=[dict(name='w16',generator='synth_divider',baseline=dict(width=16,architecture='serial'),candidate=dict(width=16,architecture='radix4'),objective='throughput')],verify_only=False)
    result=dispatch('architecture_sweep',p,store=store,cwd=tmp_path)
    assert not result['ok'] and 'clocked_ppa' not in calls
