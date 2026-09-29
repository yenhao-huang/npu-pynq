"""Analytic factorial functions and adversarial provenance, not implementation mirrors."""
from itertools import product
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.exploration_common import fingerprint
from test_physical_analysis import record


def payload(tmp_path,n=2):
    names=['architecture','storage','extra'][:n]
    cells=[]
    for bits in product((0,1),repeat=n):
        a,b=bits[:2]
        f=tmp_path/('cell'+''.join(map(str,bits))+'.sv')
        f.write_text('module dut; // '+str(bits)+'\nendmodule\n')
        rows=[]
        for r in range(3):
            row=record('candidate',f'{bits}-{r}',luts=100-10*a-20*b+5*a*b+(7*bits[2]*a*b if n==3 else 0),fmax=200+40*a+20*b-10*a*b,ii=2)
            row['source_sha256']=fingerprint([f],'dut');rows.append(row)
        cells.append(dict(levels=dict(zip(names,bits)),configuration=dict(width=32,**dict(zip(names,bits))),status='measured',files=[str(f)],top='dut',records=rows))
    return dict(factors={k:[0,1] for k in names},cells=cells)


def test_known_main_conditional_interaction_and_ii(store,tmp_path):
    data=dispatch('candidate_ablation',payload(tmp_path),store=store)['data']
    assert data['complete'] and not data['independent_statistical_samples']
    lut=data['effects']['luts']
    assert [x['mean'] for x in lut['contrasts']]==[-7.5,-17.5,5]
    assert [x['mean'] for x in lut['conditional_effects']]==[-10,-5,-20,-15]
    throughput=data['effects']['throughput_mtransactions_s']
    assert [x['mean'] for x in throughput['contrasts']]==[17.5,7.5,-5]
    assert data['scope']=='synthetic_fixture'


def test_three_way_interaction_and_zero_luts(store,tmp_path):
    p=payload(tmp_path,3)
    out=dispatch('candidate_ablation',p,store=store)['data']
    assert out['effects']['luts']['contrasts'][-1]['mean']==7
    for row in p['cells'][0]['records']: row['metrics']['luts']=0
    assert dispatch('candidate_ablation',p,store=store)['ok']


def test_failed_cell_retained_without_biased_estimates(store,tmp_path):
    p=payload(tmp_path);p['cells'][-1].update(status='failed',failure='solver timeout; no physical evidence',records=[])
    out=dispatch('candidate_ablation',p,store=store)
    assert not out['ok'] and not out['data']['complete']
    assert len(out['data']['failures'])==1 and 'effects' not in out['data']


@pytest.mark.parametrize('mutation',['missing','duplicate','geometry','level','hash','bytes','repeat','handle','build','cycles','few','failed_records'])
def test_invalid_or_confounded_design_rejected(store,tmp_path,mutation):
    p=payload(tmp_path);c=p['cells'][-1]
    if mutation=='missing': p['cells'].pop()
    if mutation=='duplicate': p['cells'][-1]=p['cells'][0]
    if mutation=='geometry': c['configuration']['width']=64
    if mutation=='level': c['configuration']['storage']=0
    if mutation=='hash': c['records'][0]['source_sha256']='f'*64
    if mutation=='bytes': Path(c['files'][0]).write_text('module dut; endmodule')
    if mutation=='repeat': c['records'].append(c['records'][-1])
    if mutation=='handle': c['records'][0]['evidence']['metrics']=p['cells'][0]['records'][0]['evidence']['metrics']
    if mutation=='build': c['records'][0]['build']='other'
    if mutation=='cycles': c['records'][0]['latency_cycles']=2
    if mutation=='few': c['records'].pop()
    if mutation=='failed_records': c.update(status='failed',failure='failed')
    with pytest.raises(InvalidInput): dispatch('candidate_ablation',p,store=store)


def test_repeat_variation_is_preserved(store,tmp_path):
    p=payload(tmp_path);p['cells'][-1]['records'][2]['metrics']['luts']+=3
    out=dispatch('candidate_ablation',p,store=store)['data']
    interaction=out['effects']['luts']['contrasts'][-1]
    assert interaction['per_repeat']==[5,5,8]
    assert interaction['min']==5 and interaction['max']==8 and interaction['mean']==6
