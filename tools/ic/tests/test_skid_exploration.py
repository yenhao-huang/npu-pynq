"""Stream ordering, capacity and bubble behavior under adversarial stalls."""
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.stream import stream_vectors


def check(store,d,**extra):
    p=dict(contract='stream',files=d['files'],top=d['top'],width=d['width'],depth=d['stages'],stream_architecture=d['architecture'],random_cycles=8192)
    p.update(extra)
    return dispatch('sequential_scoreboard',p,store=store)


@pytest.mark.parametrize('width,stages',[(32,8),(64,16),(8,2),(16,32)])
@pytest.mark.parametrize('architecture',['elastic','skid'])
def test_pipeline_oracle_and_observations(store,width,stages,architecture):
    d=dispatch('synth_skid_buffer',dict(width=width,stages=stages,architecture=architecture),store=store)['data']
    for fixture in (False,True):
        r=check(store,d,files=d['timing_files'] if fixture else d['files'],top=d['timing_top'] if fixture else d['top'],timing_fixture=fixture)
        assert r['ok'],r
        assert r['data']['calibration']['minimum_latency_cycles']==stages
        assert r['data']['calibration']['initiation_interval']==1
        assert r['data']['calibration']['capacity']==d['capacity']
        assert r['data']['source_sha256']==d['timing_sha256' if fixture else 'source_sha256']


@pytest.mark.parametrize('mutation',['reset','spill','capacity','fixture','architecture'])
def test_skid_protocol_mutations(store,tmp_path,mutation):
    d=dispatch('synth_skid_buffer',dict(width=32,stages=8),store=store)['data']
    fixture=mutation=='fixture'
    source=Path(d['timing_files'][1] if fixture else d['files'][0]).read_text()
    changed=source
    if mutation=='reset': changed=source.replace('count<=0;head<=0;', 'count<=1;head<=0;')
    if mutation=='spill': changed=source.replace('head<=spill;', 'head<=data_in;')
    if mutation=='capacity': changed=source.replace('(count<2)','(count<1)')
    if mutation=='fixture': changed=source.replace('valid_out<=core_valid;', 'valid_out<=valid_q;')
    extra={}
    if mutation=='architecture': extra['stream_architecture']='elastic'
    else: assert source!=changed
    wrong=tmp_path/'wrong.sv';wrong.write_text(changed)
    r=check(store,d,files=[d['files'][0],str(wrong)] if fixture else [str(wrong)],top=d['timing_top'] if fixture else d['top'],timing_fixture=fixture,**extra)
    assert not r['ok'] and r['data']['first_mismatch'] is not None
    assert r['data']['calibration'] is None


@pytest.mark.parametrize('architecture',['elastic','skid'])
def test_stream_model_holds_input_and_drains(architecture):
    rows,cov,cal=stream_vectors(32,8,architecture,8192,928)
    assert rows==stream_vectors(32,8,architecture,8192,928)[0]
    assert all(cov.values()) and cal['latency_samples']>=64
    for before,after in zip(rows,rows[1:]):
        if not before[0] and before[1] and not before[4] and not after[0]:
            assert after[1] and before[3]==after[3]
    assert not rows[-1][5]


@pytest.mark.parametrize('stages',[1,33])
def test_stage_bounds(store,stages):
    with pytest.raises(InvalidInput): dispatch('synth_skid_buffer',dict(stages=stages),store=store)
