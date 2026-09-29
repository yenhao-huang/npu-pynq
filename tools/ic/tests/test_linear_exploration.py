"""Independent bit-serial integer recurrences for parallel GF(2) hardware."""
import random
import shutil
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from test_arithmetic_exploration import simulate


def advance(state,width,polynomial,incoming=0):
    feedback=((state>>(width-1))&1)^incoming
    return ((state<<1)&((1<<width)-1))^(polynomial if feedback else 0)


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('architecture',['unrolled','matrix','shared'])
@pytest.mark.parametrize('width,polynomial,data_width',[(16,0x1021,32),(32,0x04C11DB7,64),(32,0x04C11DB7,128)])
def test_crc_oracle(store,tmp_path,architecture,width,polynomial,data_width):
    data=dispatch('synth_crc_parallel',dict(width=width,polynomial=polynomial,data_width=data_width,architecture=architecture),store=store)['data']
    rng=random.Random(928);inputs=width+data_width
    values=[0,(1<<inputs)-1]+[1<<bit for bit in range(inputs)]+[rng.getrandbits(inputs) for _ in range(1024)]
    expected=[]
    for value in values:
        state=value&((1<<width)-1)
        for bit in reversed(range(data_width)): state=advance(state,width,polynomial,(value>>(width+bit))&1)
        expected.append(state)
    simulate(tmp_path,data,values,expected)


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('architecture',['unrolled','matrix','shared'])
@pytest.mark.parametrize('width,polynomial,steps',[(16,0x1021,1),(32,0x04C11DB7,64),(64,0x1B,128)])
def test_jump_oracle(store,tmp_path,architecture,width,polynomial,steps):
    data=dispatch('synth_lfsr_jump',dict(width=width,polynomial=polynomial,steps=steps,architecture=architecture),store=store)['data']
    rng=random.Random(928)
    values=[0,(1<<width)-1]+[1<<bit for bit in range(width)]+[rng.getrandbits(width) for _ in range(1024)]
    expected=[]
    for value in values:
        state=value
        for _ in range(steps): state=advance(state,width,polynomial)
        expected.append(state)
    simulate(tmp_path,data,values,expected)


@pytest.mark.parametrize('op,payload',[
    ('synth_crc_parallel',dict(polynomial=2)),('synth_crc_parallel',dict(width=8,polynomial=257)),
    ('synth_lfsr_jump',dict(steps=0)),('synth_lfsr_jump',dict(steps=257)),
])
def test_linear_bounds(store,op,payload):
    with pytest.raises(InvalidInput): dispatch(op,payload,store=store)


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
def test_wrong_crc_polynomial_is_detected(store):
    a=dispatch('synth_crc_parallel',dict(data_width=64),store=store)['data']
    b=dispatch('synth_crc_parallel',dict(data_width=64,polynomial=0x1EDC6F41),store=store)['data']
    result=dispatch('vector_equivalence',dict(reference_files=[a['core_path']],candidate_files=[b['core_path']],top='dut',input_width=96,output_width=32,combinational_contract=True,random_vectors=1024),store=store)
    assert not result['ok']
