"""Logical memory, conflicts and read/write collision invariants."""
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.regfile import regfile_trace


def check(store,d,**extra):
    p=dict(contract='regfile',files=d['files'],top=d['top'],width=d['width'],depth=d['depth'],banks=d['banks'],random_cycles=8192)
    p.update(extra)
    return dispatch('sequential_scoreboard',p,store=store)


@pytest.mark.parametrize('width,depth,banks',[(16,64,2),(32,128,4),(8,16,8),(64,256,8)])
@pytest.mark.parametrize('architecture',['registers','banked'])
def test_independent_logical_memory(store,width,depth,banks,architecture):
    d=dispatch('synth_banked_regfile',dict(width=width,depth=depth,banks=banks,architecture=architecture),store=store)['data']
    for fixture in (False,True):
        r=check(store,d,files=d['timing_files'] if fixture else d['files'],top=d['timing_top'] if fixture else d['top'],timing_fixture=fixture)
        assert r['ok'],r
        assert r['data']['coverage']['written_addresses']==r['data']['coverage']['read_addresses']==depth
        assert r['data']['source_sha256']==d['timing_sha256' if fixture else 'source_sha256']


@pytest.mark.parametrize('mutation',['conflict','reset','bank_row','write_first','fixture'])
def test_memory_mutations(store,tmp_path,mutation):
    d=dispatch('synth_banked_regfile',dict(width=16,depth=64,banks=2),store=store)['data']
    fixture=mutation=='fixture'
    source=Path(d['timing_files'][1] if fixture else d['files'][0]).read_text()
    if mutation=='conflict': changed=source.replace('accept1=re1 && !conflict','accept1=re1')
    if mutation=='reset': changed=source.replace('initialized<=0',"initialized<=64'hffffffffffffffff")
    if mutation=='bank_row': changed=source.replace('ra0[5:1]','ra0[4:0]')
    if mutation=='write_first': changed=source.replace('(accept0 ? read0 :', '(accept0 ? ((we && wa==ra0) ? wd : read0) :')
    if mutation=='fixture': changed=source.replace('y<=result;',"y<=result ^ 35'd1;")
    assert source!=changed
    wrong=tmp_path/'wrong.sv';wrong.write_text(changed)
    r=check(store,d,files=[d['files'][0],str(wrong)] if fixture else [str(wrong)],top=d['timing_top'] if fixture else d['top'],timing_fixture=fixture)
    assert not r['ok'] and r['data']['first_mismatch'] is not None


@pytest.mark.parametrize('params',[dict(depth=63),dict(banks=3),dict(width=7),dict(depth=8)])
def test_invalid_storage_geometry(store,params):
    with pytest.raises(InvalidInput): dispatch('synth_banked_regfile',params,store=store)


def test_memory_stimulus_reproducibility():
    rows,coverage=regfile_trace(32,128,4,8192,928)
    assert rows==regfile_trace(32,128,4,8192,928)[0]
    assert all(coverage.values()) and coverage['read_addresses']==coverage['written_addresses']==128
