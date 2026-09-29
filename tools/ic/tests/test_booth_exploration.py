"""Signed multiplication oracle, recoding boundaries and mutation controls."""
from pathlib import Path
import random
import shutil
import pytest
from ic_core import dispatch
from test_arithmetic_exploration import simulate


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width',[8,16,17,32,64])
@pytest.mark.parametrize('architecture',['native','radix2','radix4'])
def test_booth_integer_oracle(store,tmp_path,width,architecture):
    data=dispatch('synth_booth_multiplier',dict(width=width,architecture=architecture),store=store)['data']
    mask=(1<<width)-1;outmask=(1<<(2*width))-1
    edge=[0,1,-1,2,-2,-(1<<(width-1)),(1<<(width-1))-1]
    pairs=[(a,b) for a in edge for b in edge]
    pairs += [(a,b) for a in edge for b in [((1<<k)-1) for k in range(1,width)]]
    rng=random.Random(928)
    pairs += [(rng.randrange(-(1<<(width-1)),1<<(width-1)),rng.randrange(-(1<<(width-1)),1<<(width-1))) for _ in range(2048)]
    simulate(tmp_path,data,[(a&mask)|((b&mask)<<width) for a,b in pairs],[(a*b)&outmask for a,b in pairs])


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('mutation',['sign_extension','negative_two'])
def test_booth_recoding_mutations(store,tmp_path,mutation):
    data=dispatch('synth_booth_multiplier',dict(width=16),store=store)['data']
    source=Path(data['core_path']).read_text()
    if mutation=='sign_extension': changed=source.replace('{16{a[15]}}',"16'd0")
    else: changed=source.replace("==3'b100)","==3'b111)")
    assert source!=changed
    path=tmp_path/'wrong.sv';path.write_text(changed)
    result=dispatch('vector_equivalence',dict(reference_files=[data['core_path']],candidate_files=[str(path)],top='dut',input_width=32,output_width=32,random_vectors=2048,combinational_contract=True),store=store)
    assert not result['ok']
