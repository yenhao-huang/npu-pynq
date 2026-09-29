"""Signed extrema, packing and independent integer models for arithmetic kernels."""
from pathlib import Path
import random
import re
import shutil
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from test_arithmetic_exploration import simulate


def signed(word,width):
    return word-(1<<width) if word&(1<<(width-1)) else word


def packed_vectors(width,words):
    mask=(1<<width)-1
    def pack(values): return sum((value&mask)<<(index*width) for index,value in enumerate(values))
    values=[pack([value]*words) for value in (0,1,-1,1<<(width-1),(1<<(width-1))-1)]
    values.extend(pack([(-1 if index%2 else 1)*(1<<(width-2)) for index in range(words)]) for _ in range(1))
    values.extend(pack([1 if index==active else 0 for index in range(words)]) for active in range(words))
    rng=random.Random(928)
    return values+[rng.getrandbits(width*words) for _ in range(1024)]


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width,coefficients',[(16,[3,5,7,11,11,7,5,3]),(32,[3,5,7,11,11,7,5,3]),(16,[-3,0,7,0,-3]),(8,[-32767,32767,32767,-32767])])
@pytest.mark.parametrize('architecture',['direct','symmetric','direct_serial'])
def test_fir_signed_oracle(store,tmp_path,width,coefficients,architecture):
    data=dispatch('synth_fir',dict(width=width,coefficients=coefficients,architecture=architecture),store=store)['data']
    values=packed_vectors(width,len(coefficients));mask=(1<<width)-1;outmask=(1<<data['output_width'])-1
    expected=[sum(signed((value>>(i*width))&mask,width)*c for i,c in enumerate(coefficients))&outmask for value in values]
    simulate(tmp_path,data,values,expected)
    if architecture=='symmetric': assert data['multiplier_nodes']<= (len(coefficients)+1)//2


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width,lanes',[(16,8),(32,8),(8,3),(16,7)])
@pytest.mark.parametrize('architecture',['serial','balanced','compressor'])
def test_signed_dot_oracle(store,tmp_path,width,lanes,architecture):
    data=dispatch('synth_dot_product',dict(width=width,lanes=lanes,architecture=architecture),store=store)['data']
    values=packed_vectors(width,2*lanes);mask=(1<<width)-1;outmask=(1<<data['output_width'])-1
    expected=[]
    for value in values:
        words=[signed((value>>(i*width))&mask,width) for i in range(2*lanes)]
        expected.append(sum(words[2*i]*words[2*i+1] for i in range(lanes))&outmask)
    simulate(tmp_path,data,values,expected)


@pytest.mark.parametrize('op,parameters',[
    ('synth_fir',dict(coefficients=[1,2,3])),
    ('synth_fir',dict(coefficients=[0,0,0])),
    ('synth_fir',dict(coefficients=[32768,1,32768])),
    ('synth_dot_product',dict(lanes=1)),
    ('synth_dot_product',dict(width=64)),
])
def test_datapath_contract_rejections(store,op,parameters):
    with pytest.raises(InvalidInput): dispatch(op,parameters,store=store)


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('op',['synth_fir','synth_dot_product'])
def test_arithmetic_mutations_fail(store,tmp_path,op):
    data=dispatch(op,{},store=store)['data']
    source=Path(data['core_path']).read_text()
    wrong=source.replace(' + ',' - ',1) if op=='synth_fir' else re.sub(r'\$signed\((n\d+)\)',r'\1',source)
    assert wrong!=source
    path=tmp_path/'wrong.sv';path.write_text(wrong)
    checked=dispatch('vector_equivalence',dict(reference_files=[data['core_path']],candidate_files=[str(path)],top='dut',input_width=data['input_width'],output_width=data['output_width'],combinational_contract=True,random_vectors=1024),store=store)
    assert not checked['ok']

@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width',[16,32,64])
@pytest.mark.parametrize('is_signed',[False,True])
@pytest.mark.parametrize('architecture',['dual','shared'])
def test_saturating_addsub_oracle(store,tmp_path,width,is_signed,architecture):
    data=dispatch('synth_saturating_alu',dict(width=width,signed=is_signed,architecture=architecture),store=store)['data']
    mask=(1<<width)-1
    edges=[0,1,mask,1<<(width-1),(1<<(width-1))-1,(1<<(width-1))+1]
    values=[a|(b<<width)|(subtract<<(2*width)) for a in edges for b in edges for subtract in (0,1)]
    rng=random.Random(928);values += [rng.getrandbits(2*width+1) for _ in range(2048)]
    expected=[];low=-(1<<(width-1)) if is_signed else 0;high=(1<<(width-1))-1 if is_signed else mask
    for value in values:
        a=value&mask;b=(value>>width)&mask;subtract=value>>(2*width)
        if is_signed: a=signed(a,width);b=signed(b,width)
        total=a-b if subtract else a+b
        expected.append((min(high,max(low,total))&mask)|((not low<=total<=high)<<width))
    simulate(tmp_path,data,values,expected)

@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
def test_saturating_overflow_flag_mutation(store,tmp_path):
    data=dispatch('synth_saturating_alu',dict(width=32),store=store)['data']
    source=Path(data['core_path']).read_text()
    wrong=re.sub(r'assign y=\{[^,]+,',"assign y={1'b0,",source)
    assert source!=wrong
    path=tmp_path/'wrong.sv';path.write_text(wrong)
    check=dispatch('vector_equivalence',dict(reference_files=[data['core_path']],candidate_files=[str(path)],top='dut',input_width=65,output_width=33,combinational_contract=True,random_vectors=1024),store=store)
    assert not check['ok']


def test_saturation_width_bound(store):
    with pytest.raises(InvalidInput): dispatch('synth_saturating_alu',dict(width=4),store=store)
