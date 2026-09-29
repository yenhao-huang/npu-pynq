"""Independent integer phase model for synchronous control traces."""
import hashlib
import random
import re
from ...errors import InvalidInput
from ...process import run as run_process
from ..exploration_common import sources, fingerprint
from ..synth.exploration import Result


def phase_trace(states,cycles,seed):
    rng=random.Random(seed);phase=None;rows=[];visited=set()
    coverage=dict(advance=0,hold=0,wrap=0,reset_active=0)
    for cycle in range(4*states+cycles):
        rst=int(cycle==0 or cycle==3*states or (cycle>4*states and cycle%997==0))
        advance=int(cycle<2*states or (cycle>=3*states and rng.random()<.72))
        rows.append((rst,advance,None if phase is None else 1<<phase))
        if phase is not None: visited.add(phase)
        if rst:
            coverage['reset_active']+=int(phase is not None and phase!=0);phase=0
        elif advance:
            coverage['advance']+=1;coverage['wrap']+=int(phase==states-1);phase=(phase+1)%states
        else: coverage['hold']+=1
    coverage['visited_states']=len(visited)
    return rows,coverage


def check_phase(p,ctx):
    if p.depth<4 or p.depth>256 or p.depth&(p.depth-1): raise InvalidInput('phase states must be a power of two in 4..256')
    if p.width!=1: raise InvalidInput('phase contract uses width=1 and depth=states')
    files=sources(p.files,ctx.cwd)
    if any('`include' in path.read_text() for path in files): raise InvalidInput('Inline source dependencies')
    sha=fingerprint(files,p.top)
    rows,coverage=phase_trace(p.depth,p.random_cycles,p.seed)
    delay=2 if p.timing_fixture else 0
    stimuli=rows+[(0,0,None)]*delay
    stimulus=''.join(f'{(rst<<1)|advance:x}\n' for rst,advance,_ in stimuli)
    (ctx.run.work/'phase.hex').write_text(stimulus)
    tb=ctx.run.work/'phase_tb.sv';binary=ctx.run.work/'phase.vvp'
    tb.write_text(f'''module phase_tb;
reg clk=0,rst,advance;
wire [{p.depth-1}:0] phases;
reg [1:0] vectors[0:{len(stimuli)-1}];
{p.top} dut(.clk(clk),.rst(rst),.advance(advance),.phases(phases));
integer i;
initial begin
$readmemh("phase.hex",vectors);
for(i=0;i<{len(stimuli)};i=i+1) begin
{{rst,advance}}=vectors[i];#4;$display("IC_PHASE %0d %h",i,phases);
#1;clk=1;#5;clk=0;
end
$finish;
end
endmodule
''')
    build=run_process(['iverilog','-g2012','-s','phase_tb','-o',str(binary),*map(str,files),str(tb)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'compile.log',timeout_s=p.timeout_s)
    if build.exit_code or build.timed_out or re.search(r'warning:.*(port|width|dangling)',build.text(),re.I):
        return Result(ok=False,data=dict(log=ctx.run.handle('compile.log')),note='Phase interface compilation failed.')
    run=run_process(['vvp',str(binary)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'phase.log',timeout_s=p.timeout_s)
    text=run.text(limit_bytes=len(stimuli)*(p.depth//4+100))
    observed=re.findall(r'^IC_PHASE ([0-9]+) ([0-9a-fAxXzZ]+)\s*$',text,re.M)
    if run.exit_code or run.timed_out or len(observed)!=len(stimuli) or [int(r[0]) for r in observed]!=list(range(len(stimuli))) or re.search(r'\b(ERROR|FATAL)\b',text):
        return Result(ok=False,data=dict(log=ctx.run.handle('phase.log'),timed_out=run.timed_out),note='Incomplete phase simulation.')
    mismatch=None
    for row,(_,_,expected) in zip(observed[delay:],rows):
        if expected is not None and (any(c in row[1].lower() for c in 'xz') or int(row[1],16)!=expected):
            mismatch=dict(cycle=int(row[0]),expected=expected,observed=row[1]);break
    unchanged=sha==fingerprint(files,p.top)
    passed=mismatch is None and unchanged and all(coverage.values()) and coverage['visited_states']==p.depth
    return Result(ok=passed,data=dict(passed=passed,source_sha256=sha,source_unchanged=unchanged,coverage=coverage,first_mismatch=mismatch,
        cycles=len(stimuli),observation_delay_cycles=delay,latency_cycles=1 if passed else None,initiation_interval=1 if passed else None,
        stimulus_sha256=hashlib.sha256(stimulus.encode()).hexdigest(),log=ctx.run.handle('phase.log')),
        note='Independent integer modulo-state oracle checks every post-reset output, enable hold, wrap and reset edge. Each phase is visited. Fixture observations lag by two cycles. Bounded simulation is not unbounded formal proof.')
