"""Combinational SAT equivalence using local Yosys or an existing container."""
from typing import Literal
import shutil
import json
from pathlib import Path
from pydantic import Field
from ...errors import InvalidInput
from ...process import run as run_process
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, sources, fingerprint, require_combinational
from ..synth.exploration import Result
from . import CATEGORY


# These word cells have total binary semantics. In particular, division by
# zero, out-of-range $shiftx, priority-mux ambiguity and symbolic cells cannot
# become hidden behind an abstract product. Unknown future cells fail closed.
TOTAL_BINARY_CELLS={
    '$add','$sub','$mul','$and','$or','$xor','$xnor','$not',
    '$logic_not','$logic_and','$logic_or','$eq','$ne','$eqx','$nex',
    '$lt','$le','$ge','$gt','$reduce_and','$reduce_or','$reduce_xor',
    '$reduce_xnor','$reduce_bool','$shl','$shr','$sshl','$sshr','$shift',
    '$mux','$pos','$neg','$concat','$slice',
}


class FormalIn(ExplorationInput):
    backend: str | None = Field(default='equivalence_yosys', description='Operation-specific execution backend.')
    reference_files: list[str] = Field(min_length=1, description='Reference self-contained SystemVerilog sources.')
    candidate_files: list[str] = Field(min_length=1, description='Candidate self-contained SystemVerilog sources.')
    top: Identifier = Field(description='Top module identifier, identical in separately compiled variants.')
    input_width: int = Field(ge=1, le=4096, description='Complete packed input port x width in bits.')
    output_width: int = Field(ge=1, le=4096, description='Complete packed output port y width in bits.')
    container: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$', description='Existing Docker container with Yosys; null uses local Yosys.')
    normalization: Literal['word','aig','bitwise','bitwise_products'] = Field(default='word',description='Whole-output SAT, AIG normalization, independent SAT per output bit, or per-bit SAT after making product-node outputs arbitrary (a stronger proof obligation).')
    timeout_s: int = Field(default=120, ge=1, le=1800, description='Maximum process runtime in seconds.')


@backend('debug', 'equivalence_yosys')
class FormalChecker:
    def yosys_equivalence(self, params, ctx):
        designs=[sources(params.reference_files,ctx.cwd),sources(params.candidate_files,ctx.cwd)]
        hashes=[fingerprint(files,params.top) for files in designs]
        stage=ctx.run.work/'formal'
        stage.mkdir()
        for label,files in zip(('reference','candidate'),designs):
            require_combinational(files)
            if any('`include' in p.read_text(encoding='utf-8') for p in files):
                raise InvalidInput('inline include dependencies before SAT checking')
            (stage/(label+'.sv')).write_text('\n'.join(p.read_text(encoding='utf-8') for p in files),encoding='utf-8')
        bitwise=params.normalization in ('bitwise','bitwise_products')
        match_port=f', output [{params.output_width-1}:0] matches' if bitwise else ''
        match_assign='assign matches=~(a^b);' if bitwise else ''
        (stage/'miter.sv').write_text(f'''module ic_miter(input [{params.input_width-1}:0] x, output pass{match_port});
wire [{params.output_width-1}:0] a,b;
ic_reference ref_dut(.x(x),.y(a));
ic_candidate cand_dut(.x(x),.y(b));
assign pass=(a==b);
{match_assign}
endmodule
''',encoding='utf-8')
        normalize='techmap; opt; abc -g AND; opt_clean' if params.normalization=='aig' else ''
        if params.normalization=='bitwise_products':
            normalize='select -assert-none t:$any* t:$all*; write_json original.json; opt_merge; cutpoint t:$mul; rename -witness; expose -input t:$anyseq %co1; opt_clean; check -assert'
        signals=[f'matches[{i}]' for i in range(params.output_width)] if bitwise else ['pass']
        proofs='\n'.join(f'sat -verify -prove {signal} 1 -set-def-inputs -show-inputs -show-outputs -timeout {params.timeout_s}' for signal in signals)
        script=f'''read_verilog -sv reference.sv
prep -top {params.top} -flatten
rename {params.top} ic_reference
design -stash reference
read_verilog -sv candidate.sv
prep -top {params.top} -flatten
rename {params.top} ic_candidate
design -copy-from reference ic_reference
read_verilog -sv miter.sv
prep -top ic_miter -flatten
check -assert
select -assert-none t:$*ff* t:$*latch* t:$mem*
select *
{normalize}
{proofs}
'''
        (stage/'prove.ys').write_text(script,encoding='utf-8')
        prefix=[]
        if params.container:
            if not shutil.which('docker'):
                raise InvalidInput('Docker executable unavailable')
            remote='/tmp/ic-formal-'+ctx.run.run_id
            for step,argv in [('mkdir',['docker','exec',params.container,'mkdir',remote]),('copy',['docker','cp',str(stage)+ '/.',params.container+':'+remote])]:
                result=run_process(argv,cwd=ctx.run.work,log_path=ctx.run.artifacts/(step+'.log'),timeout_s=30)
                if result.exit_code or result.timed_out:
                    return Result(ok=False,data={'log':ctx.run.handle(step+'.log')},note='Container staging failed; no proof.')
            prefix=['docker','exec','-w',remote,params.container]
        elif not shutil.which('yosys'):
            raise InvalidInput('Yosys executable unavailable; provide an existing container')
        version=run_process(prefix+['yosys','-V'],cwd=stage,log_path=ctx.run.artifacts/'version.log',timeout_s=30)
        # Docker-client termination does not stop the exec process inside the container.
        # GNU timeout owns a process group there, including any ABC child.
        guard=['timeout','--signal=TERM','--kill-after=5s',str(params.timeout_s+30)] if params.container else []
        result=run_process(prefix+guard+['yosys','-s','prove.ys'],cwd=stage,log_path=ctx.run.artifacts/'proof.log',timeout_s=params.timeout_s+60)
        text=result.text(limit_bytes=8_000_000)
        solver_timeout='ERROR: Called with -verify and proof did time out!' in text
        timed_out=result.timed_out or (bool(guard) and result.exit_code in (124,137)) or solver_timeout
        unchanged=hashes==[fingerprint(files,params.top) for files in designs]
        completed=text.count('SAT proof finished - no model found: SUCCESS!')
        abstraction_defined=True
        if params.normalization=='bitwise_products':
            if params.container:
                copied=run_process(['docker','cp',params.container+':'+remote+'/original.json',str(stage/'original.json')],cwd=stage,log_path=ctx.run.artifacts/'netlist-copy.log',timeout_s=30)
                abstraction_defined=copied.exit_code==0 and not copied.timed_out
            try:
                original=json.loads((stage/'original.json').read_text())
                # Replacing an X/Z-producing product with a defined input would
                # narrow four-state behavior. Reject such source netlists even
                # when the abstract SAT obligations happened to succeed.
                for module in original['modules'].values():
                    abstraction_defined &= all(cell['type'] in TOTAL_BINARY_CELLS for cell in module['cells'].values())
                    buses=[wire['bits'] for wire in module['netnames'].values()]
                    buses += [bits for cell in module['cells'].values() for bits in cell['connections'].values()]
                    abstraction_defined &= all(isinstance(bit,int) or bit in ('0','1') for bus in buses for bit in bus)
            except (OSError,ValueError,KeyError,TypeError):
                abstraction_defined=False
        passed=(result.exit_code==0 and not timed_out and unchanged and abstraction_defined and 'Warning:' not in text and completed==len(signals))
        return Result(ok=passed,data=dict(proved=passed,reference_sha256=hashes[0],candidate_sha256=hashes[1],engine=version.text().strip(),normalization=params.normalization,proof_obligations=len(signals),completed_obligations=completed,abstraction_defined=abstraction_defined,log=ctx.run.handle('proof.log'),exit_code=result.exit_code,timed_out=timed_out,source_unchanged=unchanged),note='Combinational Yosys SAT proof for defined binary inputs and the complete packed x/y interface. State cells are rejected; failure or timeout is not equivalence. bitwise_products merges identical cells and replaces multiplication outputs with arbitrary defined auxiliary inputs; success proves the stronger overapproximated circuit for every such value, but an abstract counterexample need not be a concrete RTL counterexample. Original X/Z constants and preexisting symbolic sources are rejected in this mode.')


CATEGORY.ops.append(Op('yosys_equivalence', FormalIn, Result, 'Prove combinational packed-interface equivalence with Yosys SAT and reject state cells.',long_running=True))
