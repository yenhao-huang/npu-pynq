"""Independent integer oracle for arithmetic generators and carry boundaries."""
import random
import shutil
import subprocess
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.synth.arithmetic import signed_digits


def simulate(tmp_path,data,values,expected):
    inputs,outputs=data['input_width'],data['output_width']
    (tmp_path/'vectors.hex').write_text(''.join(f'{x:x}\n' for x in values))
    tb=tmp_path/'tb.sv'
    tb.write_text(f'''module tb;
reg [{inputs-1}:0] x;
wire [{outputs-1}:0] y;
reg [{inputs-1}:0] vectors[0:{len(values)-1}];
dut d(.x(x),.y(y));
integer i;
initial begin
$readmemh("vectors.hex",vectors);
for(i=0;i<{len(values)};i=i+1) begin x=vectors[i]; #1; $display("VALUE %h",y); end
$finish;
end
endmodule
''')
    binary=tmp_path/'check.vvp'
    build=subprocess.run(['iverilog','-g2012','-s','tb','-o',str(binary),data['core_path'],str(tb)],capture_output=True,text=True,timeout=30)
    assert build.returncode==0,build.stderr
    result=subprocess.run(['vvp',str(binary)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    rows=[line.split()[1] for line in result.stdout.splitlines() if line.startswith('VALUE ')]
    assert len(rows)==len(values)
    assert not any(any(c in row.lower() for c in 'xz') for row in rows)
    assert [int(row,16) for row in rows]==expected


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width',[32,64])
@pytest.mark.parametrize('architecture',['native','kogge_stone','sklansky'])
def test_prefix_oracle(store,tmp_path,width,architecture):
    data=dispatch('synth_prefix_adder',dict(width=width,architecture=architecture),store=store)['data']
    rng=random.Random(928); mask=(1<<width)-1
    values=[0,(1<<(2*width+1))-1]+[(mask<<(width)) | (1<<i) for i in range(width)]+[rng.getrandbits(2*width+1) for _ in range(1024)]
    simulate(tmp_path,data,values,[(x&mask)+((x>>width)&mask)+(x>>(2*width)) for x in values])


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width,constant',[(16,255),(32,65535),(16,1),(16,32768),(16,683)])
@pytest.mark.parametrize('architecture',['native','binary','csd'])
def test_constant_oracle(store,tmp_path,width,constant,architecture):
    data=dispatch('synth_csd_multiplier',dict(width=width,constant=constant,architecture=architecture),store=store)['data']
    rng=random.Random(928)
    values=[0,(1<<width)-1]+[1<<i for i in range(width)]+[rng.getrandbits(width) for _ in range(1024)]
    simulate(tmp_path,data,values,[x*constant for x in values])


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width',[16,32])
@pytest.mark.parametrize('architecture',['independent','shared'])
def test_multiple_constants_oracle(store,tmp_path,width,architecture):
    constants=[45,51,85,255]
    data=dispatch('synth_mcm',dict(width=width,constants=constants,architecture=architecture),store=store)['data']
    rng=random.Random(928); output_word=width+max(constants).bit_length()
    values=[0,(1<<width)-1]+[rng.getrandbits(width) for _ in range(2048)]
    expected=[sum((x*c)<<(i*output_word) for i,c in enumerate(constants)) for x in values]
    simulate(tmp_path,data,values,expected)


def test_signed_digit_exactness_and_nonadjacency():
    for coefficient in range(1,65536):
        terms=signed_digits(coefficient)
        assert sum(digit*(1<<bit) for bit,digit in terms)==coefficient
        assert all(b-a>1 for (a,_),(b,_) in zip(terms,terms[1:]))


def test_reject_duplicate_coefficients(store):
    with pytest.raises(InvalidInput): dispatch('synth_mcm',dict(constants=[3,3]),store=store)

@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width,exponent',[(8,2),(16,4),(32,8),(33,5),(64,7),(128,16)])
@pytest.mark.parametrize('architecture',['native','folded'])
def test_modulo_oracle(store,tmp_path,width,exponent,architecture):
    data=dispatch('synth_constant_modulo',dict(width=width,exponent=exponent,architecture=architecture),store=store)['data']
    modulus=(1<<exponent)-1; maximum=(1<<width)-1
    rng=random.Random(928)
    values=list(range(256)) if width==8 else [0,maximum,modulus-1,modulus,modulus+1,2*modulus,maximum-1]
    values += [1<<bit for bit in range(width)]+[rng.getrandbits(width) for _ in range(2048)]
    values += [multiple*modulus+delta for multiple in (3,7,maximum//modulus) for delta in (-1,0,1) if 0<=multiple*modulus+delta<=maximum]
    simulate(tmp_path,data,values,[x%modulus for x in values])
    assert data['output_width']==exponent
    assert all(row['input_bound']>row['output_bound'] for row in data['fold_bounds'])


@pytest.mark.parametrize('parameters',[dict(width=8,exponent=8),dict(exponent=1),dict(width=129),dict(architecture='alias')])
def test_modulo_invalid_contract(store,parameters):
    with pytest.raises(InvalidInput): dispatch('synth_constant_modulo',parameters,store=store)

@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
def test_modulo_noncanonical_zero_is_detected(store,tmp_path):
    data=dispatch('synth_constant_modulo',dict(width=16,exponent=4),store=store)['data']
    source=Path(data['core_path']).read_text()
    assert ' >= ' in source
    wrong=tmp_path/'wrong.sv';wrong.write_text(source.replace(' >= ',' > '))
    checked=dispatch('vector_equivalence',dict(reference_files=[data['core_path']],candidate_files=[str(wrong)],top='dut',input_width=16,output_width=4,random_vectors=1024,combinational_contract=True),store=store)
    assert not checked['ok']
