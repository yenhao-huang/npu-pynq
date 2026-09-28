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
@pytest.mark.parametrize('normalization',['word','aig','macc','bitwise','bitwise_products','bitwise_prefix'])
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

@pytest.mark.parametrize('timed_out',[False,True])
def test_vector_failure_records_process_and_coverage(store,tmp_path,monkeypatch,timed_out):
    from ic_core.tools.debug import vectors
    from ic_core.process import CommandResult
    source=tmp_path/'dut.sv';source.write_text('module dut(input [7:0] x, output [7:0] y); assign y=x; endmodule')
    def run(argv,*,cwd,log_path,timeout_s):
        compiling=argv[0]=='iverilog'
        log_path.write_text('' if compiling else 'IC_VECTOR 0 00\n')
        return CommandResult(argv,0 if compiling else 1,0,log_path,timed_out=timed_out and not compiling)
    monkeypatch.setattr(vectors,'run_process',run)
    result=dispatch('vector_equivalence',dict(reference_files=[str(source)],candidate_files=[str(source)],top='dut',input_width=8,output_width=8,combinational_contract=True),store=store)
    assert not result['ok']
    assert result['data']['timed_out'] is timed_out
    assert result['data']['observed_vectors']==1
    assert result['data']['expected_vectors']>4096


@pytest.mark.parametrize('completed,exit_code,timed_out,passes',[(8,0,False,True),(7,0,False,False),(8,1,False,False),(8,0,True,False)])
@pytest.mark.parametrize('normalization',['bitwise','bitwise_prefix'])
def test_bitwise_requires_every_obligation_and_successful_process(store,tmp_path,monkeypatch,completed,exit_code,timed_out,passes,normalization):
    from ic_core.tools.debug import formal
    from ic_core.process import CommandResult
    source=tmp_path/'dut.sv';source.write_text('module dut(input [7:0] x, output [7:0] y); assign y=x; endmodule')
    monkeypatch.setattr(formal.shutil,'which',lambda name:name)
    def fake_run(argv,*,cwd,log_path,timeout_s):
        version='-V' in argv
        log_path.write_text('Yosys 0.23' if version else 'SAT proof finished - no model found: SUCCESS!\n'*completed)
        if not version:
            commands=[line for line in (cwd/'prove.ys').read_text().splitlines() if line.startswith('sat ')]
            assert len(commands)==8
            for bit,command in enumerate(commands):
                assert f'-prove matches[{bit}] 1 ' in command
                if normalization=='bitwise_prefix' and bit:
                    assert f"-set matches[{bit-1}:0] {bit}'b"+'1'*bit+' ' in command
                else:assert '-set matches' not in command
            for label in ('reference','candidate'):
                (cwd/(label+'.json')).write_text('{"modules":{"dut":{"netnames":{},"cells":{}}}}')
        return CommandResult(argv,0 if version else exit_code,0,log_path,timed_out=False if version else timed_out)
    monkeypatch.setattr(formal,'run_process',fake_run)
    result=dispatch('yosys_equivalence',dict(reference_files=[str(source)],candidate_files=[str(source)],top='dut',input_width=8,output_width=8,normalization=normalization),store=store)
    assert result['ok'] is passes
    assert result['data']['proof_obligations']==8
    assert result['data']['completed_obligations']==completed


@pytest.mark.skipif(not shutil.which('yosys'), reason='Local Yosys unavailable')
@pytest.mark.parametrize('expression,passes',[
    ("{8'b0,x[23:16]}+p",True),
    ("({8'b0,x[23:16]}+p)^16'h8000",False),
    ("{8'b0,x[23:16]}+(x[7:0]*x[23:16])",False),
])
def test_product_abstraction_controls(store,tmp_path,expression,passes):
    ref=tmp_path/'ref.sv';cand=tmp_path/'cand.sv'
    header='module dut(input [23:0] x,output [15:0] y); wire [15:0] p=x[7:0]*x[15:8]; '
    ref.write_text(header+"assign y=p+{8'b0,x[23:16]}; endmodule")
    cand.write_text(header+'assign y='+expression+'; endmodule')
    result=dispatch('yosys_equivalence',dict(reference_files=[str(ref)],candidate_files=[str(cand)],top='dut',input_width=24,output_width=16,normalization='bitwise_products'),store=store)
    assert result['ok'] is passes


@pytest.mark.skipif(not shutil.which('yosys'), reason='Local Yosys unavailable')
@pytest.mark.parametrize('normalization',['word','aig','macc','bitwise','bitwise_products','bitwise_prefix'])
@pytest.mark.parametrize('expression',["x[7:0]/x[7:0]","16'b0*(x[7:0]*8'bx)","x[x[7:0]]"])
def test_self_equivalence_does_not_hide_partial_semantics(store,tmp_path,normalization,expression):
    source=tmp_path/'dut.sv'
    source.write_text('module dut(input [23:0] x,output [15:0] y); assign y='+expression+'; endmodule')
    result=dispatch('yosys_equivalence',dict(reference_files=[str(source)],candidate_files=[str(source)],top='dut',input_width=24,output_width=16,normalization=normalization),store=store)
    assert not result['ok']


@pytest.mark.parametrize('bits,cell_type,expected',[(None,'$mul',False),(['x'],'$mul',False),(['z'],'$mul',False),([1,'0','1'],'$mul',True),([1],'$div',False),([1],'$shiftx',False),([1],'$pmux',False)])
@pytest.mark.parametrize('normalization',['bitwise_products','macc','aig','bitwise_prefix'])
def test_product_abstraction_requires_defined_original_netlist(store,tmp_path,monkeypatch,bits,cell_type,expected,normalization):
    import json
    from ic_core.tools.debug import formal
    from ic_core.process import CommandResult
    source=tmp_path/'dut.sv';source.write_text('module dut(input x,output y); assign y=x; endmodule')
    monkeypatch.setattr(formal.shutil,'which',lambda name:name)
    def fake_run(argv,*,cwd,log_path,timeout_s):
        if '-V' in argv:
            log_path.write_text('Yosys 0.23')
        else:
            log_path.write_text('SAT proof finished - no model found: SUCCESS!\n')
            if bits is not None:
                for label in ('reference','candidate'):
                    (cwd/(label+'.json')).write_text(json.dumps({'modules':{'dut':{'netnames':{'n':{'bits':bits}},'cells':{'c':{'type':cell_type,'connections':{'Y':bits}}}}}}))
        return CommandResult(argv,0,0,log_path)
    monkeypatch.setattr(formal,'run_process',fake_run)
    result=dispatch('yosys_equivalence',dict(reference_files=[str(source)],candidate_files=[str(source)],top='dut',input_width=1,output_width=1,normalization=normalization),store=store)
    assert result['ok'] is expected
    assert result['data']['abstraction_defined'] is expected
