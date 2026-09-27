"""Combinational SAT equivalence using local Yosys or an existing container."""
from typing import Literal
import shutil
from pathlib import Path
from pydantic import Field
from ...errors import InvalidInput
from ...process import run as run_process
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, sources, fingerprint, require_combinational
from ..synth.exploration import Result
from . import CATEGORY


class FormalIn(ExplorationInput):
    backend: str | None = Field(default='equivalence_yosys', description='Operation-specific execution backend.')
    reference_files: list[str] = Field(min_length=1, description='Reference self-contained SystemVerilog sources.')
    candidate_files: list[str] = Field(min_length=1, description='Candidate self-contained SystemVerilog sources.')
    top: Identifier = Field(description='Top module identifier, identical in separately compiled variants.')
    input_width: int = Field(ge=1, le=4096, description='Complete packed input port x width in bits.')
    output_width: int = Field(ge=1, le=4096, description='Complete packed output port y width in bits.')
    container: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$', description='Existing Docker container with Yosys; null uses local Yosys.')
    normalization: Literal['word','aig'] = Field(default='word',description='Word-level SAT or techmap/ABC AND-inverter normalization before SAT.')
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
        (stage/'miter.sv').write_text(f'''module ic_miter(input [{params.input_width-1}:0] x, output pass);
wire [{params.output_width-1}:0] a,b;
ic_reference ref_dut(.x(x),.y(a));
ic_candidate cand_dut(.x(x),.y(b));
assign pass=(a==b);
endmodule
''',encoding='utf-8')
        normalize='' if params.normalization=='word' else 'techmap; opt; abc -g AND; opt_clean'
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
sat -verify -prove pass 1 -set-def-inputs -show-inputs -show-outputs -timeout {params.timeout_s}
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
        unchanged=hashes==[fingerprint(files,params.top) for files in designs]
        passed=(result.exit_code==0 and not result.timed_out and unchanged and 'Warning:' not in text and 'SAT proof finished - no model found: SUCCESS!' in text)
        return Result(ok=passed,data=dict(proved=passed,reference_sha256=hashes[0],candidate_sha256=hashes[1],engine=version.text().strip(),normalization=params.normalization,log=ctx.run.handle('proof.log'),exit_code=result.exit_code,timed_out=result.timed_out or (bool(guard) and result.exit_code in (124,137)),source_unchanged=unchanged),note='Combinational Yosys SAT proof for defined binary inputs and the complete packed x/y interface. State cells are rejected; failure or timeout is not equivalence.')


CATEGORY.ops.append(Op('yosys_equivalence', FormalIn, Result, 'Prove combinational packed-interface equivalence with Yosys SAT and reject state cells.',long_running=True))
