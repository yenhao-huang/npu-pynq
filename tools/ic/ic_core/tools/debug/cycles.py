"""Independent transaction model and cycle-performance validation for arithmetic."""
import hashlib
import random
import re
from typing import Literal
from pydantic import Field, model_validator
from ...errors import InvalidInput
from ...process import run as run_process
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, fingerprint, sources
from ..synth.exploration import Result
from . import CATEGORY


class CyclesIn(ExplorationInput):
    backend: str | None = Field(default='arithmetic_cycles',description='Independent single-outstanding arithmetic transaction checker.')
    files: list[str] = Field(min_length=1,description='Self-contained operator or registered observation-fixture RTL.')
    top: Identifier = Field(default='arithmetic_dut',description='Ready/valid scalar or matrix module with packed data ports.')
    width: int = Field(ge=8,le=64,description='Element width: unsigned scalar arithmetic or signed matrix entries (matrix maximum 32).')
    size: int = Field(default=2,ge=2,le=4,description='Square matrix dimension, used only for matrix operation.')
    operation: Literal['multiply','divide','matrix'] = Field(description='Independent Python integer or signed matrix oracle.')
    latency_cycles: int = Field(ge=1,le=129,description='Declared acceptance-to-first-valid latency to verify.')
    expected_ii: int = Field(ge=1,le=129,description='Declared sustained acceptance interval under no stalls.')
    timing_fixture: bool = Field(default=False,description='Compare the standard two-cycle delayed observation fixture, not an external handshake adapter.')
    random_cycles: int = Field(default=8192,ge=512,le=100000,description='Seeded stress cycles after directed and sustained-throughput phases.')
    seed: int = Field(default=928,ge=0,le=2**32-1,description='Deterministic stimulus seed.')
    timeout_s: int = Field(default=120,ge=1,le=600,description='Bounded compile/simulation duration per process.')


    @model_validator(mode='after')
    def matrix_width(self):
        if self.operation=='matrix' and self.width>32: raise ValueError('Matrix width is limited to 32 bits')
        return self


def arithmetic_trace(width,operation,latency,random_cycles,seed):
    mask=(1<<width)-1
    edges=[0,1,2,mask,mask-1,1<<(width-1)]
    transactions=[a|(b<<width) for a in edges for b in edges]
    def oracle(data):
        a=data&mask;b=(data>>width)&mask
        value=a*b if operation=='multiply' else ((a if b==0 else a%b)<<width)|(mask if b==0 else a//b)
        return value,a==0 or b==0
    return transaction_trace(2*width,transactions,oracle,latency,random_cycles,seed)


def transaction_trace(input_bits,transactions,oracle,latency,random_cycles,seed):
    rng=random.Random(seed)
    calibration_end=64*latency+3;stall_end=calibration_end+latency+8
    random_end=stall_end+4+random_cycles;total=random_end+2*latency+4
    job=None;pending=None;index=0;vectors=[];accepted=[];latencies=[]
    coverage=dict(accepted=0,completed=0,output_stall=0,reset_pending=0,busy_stall=0,zero_operand=0)
    for cycle in range(total):
        rst=int(cycle in (0,stall_end,stall_end+2) or (stall_end+3<cycle<random_end and cycle%997==0))
        ready=0 if calibration_end<=cycle<stall_end else (1 if cycle<stall_end+4 or cycle>=random_end else int(rng.random()<.65))
        valid=int(cycle<random_end and (cycle<stall_end+4 or rng.random()<.85))
        if pending is not None: valid=1
        data=pending if pending is not None else (transactions[index] if index<len(transactions) else rng.getrandbits(input_bits))
        out_valid=int(job is not None and job['due']<=cycle and not rst)
        in_ready=int(not rst and (job is None or (out_valid and ready)))
        output=job['value'] if job is not None else None
        vectors.append((rst,valid,ready,data,in_ready,out_valid,output))
        if rst:
            coverage['reset_pending']+=int(job is not None);job=None;pending=None
            continue
        if out_valid and not job['observed']:
            latencies.append(cycle-job['accepted']);job['observed']=True
        if out_valid and not ready: coverage['output_stall']+=1
        if valid and not in_ready and job is not None and not out_valid: coverage['busy_stall']+=1
        if out_valid and ready: coverage['completed']+=1;job=None
        if valid and in_ready:
            value,zero=oracle(data)
            job=dict(value=value,due=cycle+latency,accepted=cycle,observed=False)
            coverage['accepted']+=1;coverage['zero_operand']+=int(zero);index+=1
            if cycle<calibration_end: accepted.append(cycle)
        pending=data if valid and not in_ready else None
    if job is not None or pending is not None: raise InvalidInput('Arithmetic trace failed to drain')
    return vectors,coverage,accepted,latencies


@backend('debug','arithmetic_cycles',requires='iverilog',version_cmd=['iverilog','-V'])
class Cycles:
    def latency_throughput(self,p,ctx):
        files=sources(p.files,ctx.cwd)
        if any('`include' in path.read_text(encoding='utf-8') for path in files): raise InvalidInput('Inline includes for complete source identity')
        sha=fingerprint(files,p.top)
        if p.operation=='matrix':
            from .matrix import matrix_trace
            vectors,coverage,accepted,latencies=matrix_trace(p.width,p.size,p.latency_cycles,p.random_cycles,p.seed)
            bits=2*p.size*p.size*p.width
            output_bits=p.size*p.size*(2*p.width+(p.size-1).bit_length())
        else:
            vectors,coverage,accepted,latencies=arithmetic_trace(p.width,p.operation,p.latency_cycles,p.random_cycles,p.seed)
            bits=output_bits=2*p.width
        original=vectors;delay=2 if p.timing_fixture else 0
        vectors=vectors+[(1,0,1,0,0,0,None)]*delay
        stimulus=''.join(f'{(rst<<(bits+2))|(valid<<(bits+1))|(ready<<bits)|data:x}\n' for rst,valid,ready,data,*_ in vectors)
        (ctx.run.work/'stimulus.hex').write_text(stimulus,encoding='ascii')
        tb=ctx.run.work/'cycles.sv'
        tb.write_text(f'''module ic_arithmetic_tb;
reg clk=0,rst,valid_in,ready_out;
reg [{bits-1}:0] data_in;
wire ready_in,valid_out;
wire [{output_bits-1}:0] data_out;
reg [{bits+2}:0] stimulus[0:{len(vectors)-1}];
{p.top} dut(.clk(clk),.rst(rst),.valid_in(valid_in),.ready_in(ready_in),.data_in(data_in),.valid_out(valid_out),.ready_out(ready_out),.data_out(data_out));
integer i;
initial begin
$readmemh("stimulus.hex",stimulus);
for(i=0;i<{len(vectors)};i=i+1) begin
{{rst,valid_in,ready_out,data_in}}=stimulus[i];#4;
$display("IC_CYCLE %0d %b %b %h",i,ready_in,valid_out,data_out);
#1;clk=1;#5;clk=0;
end
$finish;
end
endmodule
''',encoding='utf-8')
        binary=ctx.run.work/'cycles.vvp'
        build=run_process(['iverilog','-g2012','-s','ic_arithmetic_tb','-o',str(binary),*map(str,files),str(tb)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'compile.log',timeout_s=p.timeout_s)
        if build.exit_code or build.timed_out or re.search(r'warning:.*(port|width|dangling)',build.text(),re.I):
            return Result(ok=False,data=dict(log=ctx.run.handle('compile.log'),timed_out=build.timed_out),note='Compilation or interface check failed.')
        run=run_process(['vvp',str(binary)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'cycles.log',timeout_s=p.timeout_s)
        text=run.text(limit_bytes=len(vectors)*(output_bits//4+100))
        rows=re.findall(r'^IC_CYCLE ([0-9]+) ([01xz]) ([01xz]) ([0-9a-fAxXzZ]+)\s*$',text,re.M)
        if run.exit_code or run.timed_out or len(rows)!=len(vectors) or [int(row[0]) for row in rows]!=list(range(len(vectors))) or re.search(r'\b(ERROR|FATAL)\b',text):
            return Result(ok=False,data=dict(log=ctx.run.handle('cycles.log'),timed_out=run.timed_out),note='Incomplete simulation; no cycle or throughput verdict.')
        mismatch=None
        for row,vector in zip(rows[delay:],original):
            expected_ready,expected_valid,expected_data=vector[4:]
            if row[1]!=str(expected_ready) or row[2]!=str(expected_valid) or (expected_valid and (any(c in row[3].lower() for c in 'xz') or int(row[3],16)!=expected_data)):
                mismatch=dict(cycle=int(row[0]),observed=dict(ready=row[1],valid=row[2],data=row[3]),expected=dict(ready=expected_ready,valid=expected_valid,data=expected_data));break
        intervals=[b-a for a,b in zip(accepted,accepted[1:])]
        covered=all(value>0 for key,value in coverage.items() if key!='busy_stall' or p.latency_cycles>1)
        unchanged=sha==fingerprint(files,p.top)
        passed=mismatch is None and unchanged and covered and len(intervals)>=32 and all(value==p.expected_ii for value in intervals)
        return Result(ok=passed,data=dict(source_sha256=sha,source_unchanged=unchanged,passed=passed,cycles=len(vectors),coverage=coverage,first_mismatch=mismatch,
            verified_latency_cycles=p.latency_cycles if passed else None,verified_initiation_interval=p.expected_ii if passed else None,
            latency_range=[min(latencies),max(latencies)] if passed and latencies else None,steady_state_acceptance_intervals=sorted(set(intervals)) if passed else [],
            observation_delay_cycles=delay,stimulus_sha256=hashlib.sha256(stimulus.encode()).hexdigest(),log=ctx.run.handle('cycles.log')),
            note='Independent exact-integer transaction oracle checks every ready/valid cycle, held outputs, reset cancellation, result packing and declared first-valid latency/II. Reported cycles require a matching full trace. Fixture mode checks delayed observations, not an external protocol adapter. Bounded simulation is not unbounded sequential formal proof.')


CATEGORY.ops.append(Op('latency_throughput',CyclesIn,Result,'Verify arithmetic latency and sustained initiation interval against independent transaction semantics.',long_running=True))
