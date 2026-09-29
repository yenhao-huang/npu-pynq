"""Cycle-exact ready/valid FIFO scoreboard with an independent Python queue."""
from collections import deque
from typing import Literal
import hashlib
import json
import random
import re
from pydantic import Field
from ...registry import Op, backend
from ...process import run as run_process
from ...errors import InvalidInput
from ..exploration_common import ExplorationInput, Identifier, fingerprint, sources
from ..synth.exploration import Result
from . import CATEGORY


class SequentialIn(ExplorationInput):
    contract: Literal['fifo','phase','regfile','stream'] = Field(default='fifo',description='FIFO, phase, regfile or elastic/skid pipeline; stream uses depth=stages.')
    stream_architecture: Literal['elastic','skid'] = Field(default='skid',description='Stream stage policy; ignored for other contracts.')
    banks: int = Field(default=2,ge=2,le=8,description='Register-file bank count; unused for FIFO and phase contracts.')
    backend: str | None = Field(default='fifo_scoreboard',description='Icarus cycle-exact sequential contract checker.')
    files: list[str] = Field(min_length=1,description='Self-contained sequential RTL sources.')
    top: Identifier = Field(default='fifo_dut',description='FIFO ready/valid top, phase clk/rst/advance/phases, or regfile clk/rst/x/y.')
    width: int = Field(ge=1,le=64,description='FIFO/register-file word width; must be one for phase contract.')
    depth: int = Field(ge=2,le=1024,description='FIFO capacity, phase count (4..256), or register-file words (16..256); phase/regfile require powers of two.')
    timing_fixture: bool = Field(default=False,description='Check the standardized registered timing fixture with two-cycle delayed observations; not an external FIFO protocol adapter.')
    random_cycles: int = Field(default=8192,ge=64,le=100000,description='Seeded cycles after fill/drain and simultaneous-transfer phases.')
    seed: int = Field(default=928,ge=0,le=2**32-1,description='Deterministic stimulus seed.')
    timeout_s: float = Field(default=120,gt=0,le=600,allow_inf_nan=False,description='Compile and simulation timeout per process, seconds.')


def fifo_vectors(width,depth,cycles,seed):
    queue=deque()
    rng=random.Random(seed)
    pending=None
    vectors=[]
    coverage=dict(push=0,pop=0,simultaneous=0,full_stall=0,empty=0,reset_nonempty=0,output_stall=0)
    for cycle in range(4*depth+cycles+2):
        rst=int(cycle==0 or cycle==3*depth or (cycle>4*depth and cycle%997==0))
        if cycle<=2*depth:
            valid,ready=1,0
        elif cycle<=3*depth:
            valid,ready=0,1
        elif cycle<=4*depth:
            valid,ready=1,1
        else:
            valid,ready=int(rng.random()<.8),int(rng.random()<.6)
        if pending is not None: valid=1
        data=pending if pending is not None else rng.getrandbits(width)
        if rst:
            expected_ready=expected_valid=0
        else:
            expected_valid=int(bool(queue))
            expected_ready=int(len(queue)<depth or (bool(queue) and ready))
        expected_data=queue[0] if queue else None
        vectors.append((rst,valid,ready,data,expected_ready,expected_valid,expected_data))
        if rst:
            coverage['reset_nonempty']+=int(bool(queue));queue.clear();pending=None
        else:
            push=valid and expected_ready
            pop=ready and expected_valid
            coverage['push']+=int(push);coverage['pop']+=int(pop)
            coverage['simultaneous']+=int(push and pop)
            coverage['full_stall']+=int(not expected_ready and valid)
            coverage['empty']+=int(not queue)
            coverage['output_stall']+=int(expected_valid and not ready)
            if pop: queue.popleft()
            if push: queue.append(data)
            pending=data if valid and not expected_ready else None
    return vectors,coverage


@backend('debug','fifo_scoreboard',requires='iverilog',version_cmd=['iverilog','-V'])
class Scoreboard:
    def sequential_scoreboard(self,p,ctx):
        if p.contract=='regfile':
            from .regfile import check_regfile
            return check_regfile(p,ctx)
        if p.contract=='phase':
            from .phase import check_phase
            return check_phase(p,ctx)
        files=sources(p.files,ctx.cwd)
        if any('`include' in path.read_text(encoding='utf-8') for path in files):
            raise InvalidInput('Inline includes for complete sequential source identity')
        sha=fingerprint(files,p.top)
        calibration=None
        if p.contract=='stream':
            from .stream import stream_vectors
            vectors,coverage,calibration=stream_vectors(p.width,p.depth,p.stream_architecture,p.random_cycles,p.seed)
        else:
            vectors,coverage=fifo_vectors(p.width,p.depth,p.random_cycles,p.seed)
        expected_vectors=vectors
        delay=2 if p.timing_fixture else 0
        vectors=vectors+[(1,0,0,0,0,0,None)]*delay
        # Input words pack reset, valid and ready above the data word.
        stimulus=''.join(f'{(rst<<(p.width+2))|(valid<<(p.width+1))|(ready<<p.width)|data:x}\n' for rst,valid,ready,data,*_ in vectors)
        (ctx.run.work/'stimulus.hex').write_text(stimulus,encoding='ascii')
        tb=ctx.run.work/'scoreboard.sv'
        tb.write_text(f'''module ic_fifo_tb;
reg clk=0, rst, valid_in, ready_out;
reg [{p.width-1}:0] data_in;
wire ready_in,valid_out;
wire [{p.width-1}:0] data_out;
reg [{p.width+2}:0] stimulus[0:{len(vectors)-1}];
{p.top} dut(.clk(clk),.rst(rst),.valid_in(valid_in),.ready_in(ready_in),.data_in(data_in),.valid_out(valid_out),.ready_out(ready_out),.data_out(data_out));
integer i;
initial begin
$readmemh("stimulus.hex",stimulus);
for(i=0;i<{len(vectors)};i=i+1) begin
{{rst,valid_in,ready_out,data_in}}=stimulus[i];
#4;
$display("IC_CYCLE %0d %b %b %h",i,ready_in,valid_out,data_out);
#1; clk=1; #5; clk=0;
end
$finish;
end
endmodule
''',encoding='utf-8')
        binary=ctx.run.work/'scoreboard.vvp'
        built=run_process(['iverilog','-g2012','-s','ic_fifo_tb','-o',str(binary),*map(str,files),str(tb)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'compile.log',timeout_s=p.timeout_s)
        if built.exit_code or built.timed_out or re.search(r'warning:.*(port|width|dangling)',built.text(),re.I):
            return Result(ok=False,data={'log':ctx.run.handle('compile.log')},note='Compile failure or interface mismatch; no protocol verdict.')
        run=run_process(['vvp',str(binary)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'cycles.log',timeout_s=p.timeout_s)
        text=run.text(limit_bytes=len(vectors)*(p.width//4+100))
        rows=re.findall(r'^IC_CYCLE ([0-9]+) ([01xz]) ([01xz]) ([0-9a-fAxXzZ]+)\s*$',text,re.M)
        if (run.exit_code or run.timed_out or re.search(r'\b(ERROR|FATAL)\b',text)
                or len(rows)!=len(vectors) or [int(row[0]) for row in rows]!=list(range(len(vectors)))):
            return Result(ok=False,data={'log':ctx.run.handle('cycles.log')},note='Incomplete simulation; no protocol verdict.')
        mismatch=None
        for i,(row,vector) in enumerate(zip(rows[delay:],expected_vectors),start=delay):
            _,ready,valid,data=row
            expected_ready,expected_valid,expected_data=vector[4:]
            if ready!=str(expected_ready) or valid!=str(expected_valid) or (expected_valid and (any(c in data.lower() for c in 'xz') or int(data,16)!=expected_data)):
                mismatch=dict(cycle=i,observed=dict(ready=ready,valid=valid,data=data),expected=dict(ready=expected_ready,valid=expected_valid,data=expected_data));break
        unchanged=sha==fingerprint(files,p.top)
        passed=mismatch is None and unchanged and all(coverage.values())
        return Result(ok=passed,data=dict(passed=passed,source_sha256=sha,source_unchanged=unchanged,stimulus_sha256=hashlib.sha256(stimulus.encode()).hexdigest(),cycles=len(vectors),observation_delay_cycles=delay,coverage=coverage,calibration=calibration if passed else None,first_mismatch=mismatch,log=ctx.run.handle('cycles.log')),note='Timing-fixture mode compares two-cycle delayed observations of a predetermined queue trace; it is not an external ready/valid adapter. Queue-model simulation checks capacity, ordering, stalls and reset flush. Stream calibration verifies no-stall latency and II only after the entire trace matches. Invalid data is ignored. This is a bounded protocol test, not unbounded sequential equivalence.')


CATEGORY.ops.append(Op('sequential_scoreboard',SequentialIn,Result,'Verify FIFO, stream, phase and banked-memory contracts with independent cycle models.',long_running=True))
