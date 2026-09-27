"""Independent integer-oracle checks for generated RTL, not string snapshots."""
import random
import shutil
import subprocess
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput

OPS=['synth_popcount','synth_priority_encoder','synth_leading_zero','synth_barrel_shifter','synth_onehot_mux','synth_argmax_tree']


def oracle(op,width,lanes,x):
    mask=(1<<width)-1
    if op=='synth_popcount': return x.bit_count()
    if op=='synth_priority_encoder': return (width | (x.bit_length()-1)) if x else 0
    if op=='synth_leading_zero': return width-x.bit_length()
    if op=='synth_barrel_shifter': return ((x&mask) << (x>>width)) & mask
    values=[(x>>(i*width)) & mask for i in range(lanes)]
    if op=='synth_onehot_mux':
        select=x>>(width*lanes)
        result=0
        for i,value in enumerate(values):
            if (select>>i)&1: result |= value
        return result
    maximum=max(values)
    return (maximum<<(lanes-1).bit_length()) | values.index(maximum)


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('scale',[0,1])
@pytest.mark.parametrize('architecture',['linear','tree'])
def test_network_oracle(store,tmp_path,op,scale,architecture):
    width=(16<<scale) if op in OPS[3:] else (64<<scale)
    lanes=16
    params=dict(width=width,architecture=architecture)
    if op in OPS[4:]: params['lanes']=lanes
    data=dispatch(op,params,store=store)['data']
    inputs,outputs=data['input_width'],data['output_width']
    rng=random.Random(928)
    values=[0,(1<<inputs)-1]+[1<<i for i in range(inputs)]+[rng.getrandbits(inputs) for _ in range(1024)]
    if op=='synth_argmax_tree':
        values += [sum(((1<<width)-1)<<(i*width) for i in pair) for pair in [(0,1),(0,15),(7,8)]]
    if op=='synth_onehot_mux':
        values += [((1<<(width*lanes))-1) | (1<<(width*lanes+i)) for i in range(lanes)]
    (tmp_path/'vectors.hex').write_text(''.join(f'{x:x}\n' for x in values))
    (tmp_path/'expected.hex').write_text(''.join(f'{oracle(op,width,lanes,x):x}\n' for x in values))
    bench=tmp_path/'oracle_tb.sv'
    bench.write_text(f'''module oracle_tb;
reg [{inputs-1}:0] x;
wire [{outputs-1}:0] y;
reg [{inputs-1}:0] stimulus[0:{len(values)-1}];
reg [{outputs-1}:0] expected[0:{len(values)-1}];
dut d(.x(x),.y(y));
integer i;
initial begin
$readmemh("vectors.hex",stimulus);
$readmemh("expected.hex",expected);
for(i=0;i<{len(values)};i=i+1) begin
x=stimulus[i]; #1;
if(y !== expected[i]) $fatal(1,"oracle mismatch %d: %h != %h",i,y,expected[i]);
end
$display("ORACLE_PASS {len(values)}");
$finish;
end
endmodule
''')
    binary=tmp_path/'oracle.vvp'
    build=subprocess.run(['iverilog','-g2012','-s','oracle_tb','-o',str(binary),data['core_path'],str(bench)],capture_output=True,text=True,timeout=30)
    assert build.returncode==0,build.stderr
    result=subprocess.run(['vvp',str(binary)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert f'ORACLE_PASS {len(values)}' in result.stdout


@pytest.mark.parametrize('op',OPS)
def test_unsupported_size_rejected(store,op):
    with pytest.raises(InvalidInput):
        dispatch(op,dict(width=17),store=store)


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('scale',[0,1])
def test_leading_zero_binary_search_oracle(store,tmp_path,scale):
    test_network_oracle(store,tmp_path,'synth_leading_zero',scale,'binary_search')
