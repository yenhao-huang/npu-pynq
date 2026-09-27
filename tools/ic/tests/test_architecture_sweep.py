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
