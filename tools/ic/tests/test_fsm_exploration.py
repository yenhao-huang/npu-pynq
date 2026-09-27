"""Phase-controller encoding and independent synchronous behavior checks."""
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.phase import phase_trace


def check(store,d,**extra):
    p=dict(contract='phase',files=d['files'],top=d['top'],width=1,depth=d['states'],random_cycles=8192)
    p.update(extra)
    return dispatch('sequential_scoreboard',p,store=store)


@pytest.mark.parametrize('states',[4,64,128,256])
@pytest.mark.parametrize('encoding',['binary','onehot'])
def test_phase_oracle_core_and_fixture(store,states,encoding):
    d=dispatch('synth_fsm',dict(states=states,encoding=encoding),store=store)['data']
    for fixture in (False,True):
        r=check(store,d,files=d['timing_files'] if fixture else d['files'],top=d['timing_top'] if fixture else d['top'],timing_fixture=fixture)
        assert r['ok'],r
        assert r['data']['coverage']['visited_states']==states
        assert r['data']['source_sha256']==d['timing_sha256' if fixture else 'source_sha256']


@pytest.mark.parametrize('mutation',['reset','hold','direction','fixture'])
def test_phase_negative_controls(store,tmp_path,mutation):
    d=dispatch('synth_fsm',dict(states=64),store=store)['data']
    fixture=mutation=='fixture'
    source=Path(d['timing_files'][1] if fixture else d['files'][0]).read_text()
    if mutation=='reset': source=source.replace("state<=64'd1", "state<=64'd2")
    if mutation=='hold': source=source.replace('else if(advance)', 'else')
    if mutation=='direction': source=source.replace('{state[62:0],state[63]}','{state[0],state[63:1]}')
    if mutation=='fixture': source=source.replace('phases<=core_phases;', "phases<=core_phases ^ 64'd1;")
    wrong=tmp_path/'wrong.sv';wrong.write_text(source)
    r=check(store,d,files=[d['files'][0],str(wrong)] if fixture else [str(wrong)],top=d['timing_top'] if fixture else d['top'],timing_fixture=fixture)
    assert not r['ok'] and r['data']['first_mismatch'] is not None


def test_phase_input_boundaries(store):
    with pytest.raises(InvalidInput): dispatch('synth_fsm',dict(states=63),store=store)


def test_phase_oracle_coverage():
    rows,cov=phase_trace(128,8192,928)
    assert rows==phase_trace(128,8192,928)[0]
    assert all(cov.values()) and cov['visited_states']==128
    for before,after in zip(rows[1:],rows[2:]):
        if before[0]: assert after[2]==1
        elif not before[1]: assert after[2]==before[2]
