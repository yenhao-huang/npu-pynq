"""Wide-interface integration coverage, including deliberate incorrect RTL."""
import shutil
import pytest
from ic_core import dispatch
from ic_core.tools.debug.vectors import stimuli, VectorIn


def test_stimuli_reproducible_and_boundaries():
    a = stimuli(128, 2048, 928)
    assert a == stimuli(128, 2048, 928)
    assert a != stimuli(128, 2048, 929)
    assert len(a) == 2436
    assert 0 in a and 2**128-1 in a and 2**127-1 in a


def test_contract_required():
    with pytest.raises(ValueError):
        VectorIn(reference_files=['a'], candidate_files=['b'], top='dut', input_width=32, output_width=32, combinational_contract=False)


@pytest.mark.skipif(not shutil.which('iverilog'), reason='Icarus unavailable')
@pytest.mark.parametrize('expression,passes', [('x[31:0]+x[63:32]', True), ('x[31:0]-x[63:32]', False), ("32'bx", False)])
def test_wide_check(store, tmp_path, expression, passes):
    ref = tmp_path/'ref.sv'
    cand = tmp_path/'cand.sv'
    ref.write_text('module dut(input [63:0] x, output [31:0] y); assign y=x[31:0]+x[63:32]; endmodule')
    cand.write_text(f'module dut(input [63:0] x, output [31:0] y); assign y={expression}; endmodule')
    result = dispatch('vector_equivalence', dict(reference_files=[str(ref)], candidate_files=[str(cand)], top='dut', input_width=64, output_width=32, random_vectors=2048, combinational_contract=True), store=store)
    assert result['ok'] is passes
    assert result['data']['vectors']==2244
    assert (result['data']['first_mismatch'] is None) is passes


def test_utilization_uses_used_not_available():
    from ic_core.tools.synth.clocked import utilization
    parsed = utilization('''| Slice LUTs* | 120 | 0 | 53200 | 0.23 |
| Slice Registers | 64 | 0 | 106400 | 0.06 |
| Block RAM Tile | 0.5 | 0 | 140 | 0.36 |
| DSPs | 2 | 0 | 220 | 0.91 |''')
    assert parsed == dict(luts=120, ffs=64, brams=.5, dsps=2)
    with pytest.raises(ValueError):
        utilization('| Slice LUTs | 120 | 53200 |')


@pytest.mark.skipif(not shutil.which('iverilog'), reason='Icarus unavailable')
@pytest.mark.parametrize('width,lanes', [(16,16),(32,16),(8,3),(8,7)])
def test_reduction_architectures(store, width, lanes):
    designs = {}
    for architecture in ['serial','balanced','compressor']:
        out=dispatch('synth_adder_tree',dict(width=width,lanes=lanes,architecture=architecture),store=store)
        designs[architecture]=out['data']
        from pathlib import Path
        assert Path(out['data']['core_path']).exists()
        assert Path(out['data']['wrapper_path']).exists()
    for architecture in ['balanced','compressor']:
        checked=dispatch('vector_equivalence',dict(reference_files=[designs['serial']['core_path']],candidate_files=[designs[architecture]['core_path']],top='dut',input_width=width*lanes,output_width=width,random_vectors=4096,combinational_contract=True),store=store)
        assert checked['ok'], checked


@pytest.mark.skipif(not shutil.which('yosys'), reason='Local Yosys unavailable')
@pytest.mark.parametrize('normalization',['word','aig'])
@pytest.mark.parametrize('expression,passes', [('x[15:0]+x[31:16]', True), ('x[15:0]-x[31:16]', False)])
def test_sat_controls(store, tmp_path, expression, passes, normalization):
    ref=tmp_path/'ref.sv'
    cand=tmp_path/'cand.sv'
    ref.write_text('module dut(input [31:0] x, output [15:0] y); assign y=x[15:0]+x[31:16]; endmodule')
    cand.write_text(f'module dut(input [31:0] x, output [15:0] y); assign y={expression}; endmodule')
    result=dispatch('yosys_equivalence',dict(reference_files=[str(ref)],candidate_files=[str(cand)],top='dut',input_width=32,output_width=16,normalization=normalization),store=store)
    assert result['ok'] is passes


@pytest.mark.skipif(not shutil.which('iverilog'), reason='Icarus unavailable')
def test_vector_rejects_truncated_interface(store, tmp_path):
    source=tmp_path/'dut.sv'
    source.write_text('module dut(input [63:0] x, output [31:0] y); assign y=x[31:0]; endmodule')
    result=dispatch('vector_equivalence',dict(reference_files=[str(source)],candidate_files=[str(source)],top='dut',input_width=32,output_width=32,combinational_contract=True),store=store)
    assert not result['ok']

@pytest.mark.parametrize('message,timed_out',[
    ('ERROR: Called with -verify and proof did time out!',True),
    ('ERROR: Called with -verify and proof did fail!',False),
])
def test_sat_internal_timeout_classification(store,tmp_path,monkeypatch,message,timed_out):
    from ic_core.tools.debug import formal
    from ic_core.process import CommandResult
    source=tmp_path/'dut.sv'
    source.write_text('module dut(input [7:0] x, output [7:0] y); assign y=x; endmodule')
    monkeypatch.setattr(formal.shutil,'which',lambda name:name)
    def fake_run(argv,*,cwd,log_path,timeout_s):
        version='-V' in argv
        log_path.write_text('Yosys 0.23' if version else message)
        return CommandResult(argv,0 if version else 1,0,log_path)
    monkeypatch.setattr(formal,'run_process',fake_run)
    result=dispatch('yosys_equivalence',dict(reference_files=[str(source)],candidate_files=[str(source)],top='dut',input_width=8,output_width=8),store=store)
    assert result['data']['timed_out'] is timed_out
    assert not result['ok']
