"""Independent logical-memory and bank arbitration oracle."""
import hashlib
import random
import re
from ...errors import InvalidInput
from ...process import run as run_process
from ..exploration_common import sources,fingerprint
from ..synth.exploration import Result


def regfile_trace(width,depth,banks,cycles,seed):
    rng=random.Random(seed);a=(depth-1).bit_length();mask=(1<<width)-1
    memory=[0]*depth;output=None;rows=[];written=set();read=set()
    coverage=dict(write=0,read0=0,read1=0,conflict=0,read_write_collision=0,read1_only=0,idle=0,reset_nonzero=0)
    end=5*depth+2+cycles
    for cycle in range(end+2):
        rst=int(cycle in (0,4*depth+1) or (5*depth+2<=cycle<end and cycle%997==0))
        wa=rng.randrange(depth);ra0=rng.randrange(depth);ra1=rng.randrange(depth);wd=rng.getrandbits(width)
        we,re0,re1=(int(rng.random()<prob) for prob in (.65,.8,.7))
        if 1<=cycle<=depth:
            wa=ra0=cycle-1;ra1=ra0^1;wd=(cycle*37)&mask;we=re0=re1=1
        elif depth<cycle<=2*depth:
            ra0=(cycle-1)%depth;ra1=(ra0+banks)%depth;we=0;re0=re1=1
        elif 2*depth<cycle<=3*depth:
            ra1=(cycle-1)%depth;we=re0=0;re1=1
        elif 3*depth<cycle<=4*depth:
            ra0=(cycle-1)%depth;ra1=ra0^1;we=0;re0=re1=1
        elif 4*depth+1<cycle<=5*depth+1:
            ra0=(cycle-2)%depth;ra1=ra0^1;we=0;re0=re1=1
        elif cycle>=end: we=re0=re1=0
        x=wd|(wa<<width)|(ra0<<(width+a))|(ra1<<(width+2*a))|(we<<(width+3*a))|(re0<<(width+3*a+1))|(re1<<(width+3*a+2))
        rows.append((rst,x,output))
        if rst:
            coverage['reset_nonzero']+=int(any(memory));memory=[0]*depth;output=0;continue
        conflict=int(re0 and re1 and ra0%banks==ra1%banks);v0=re0;v1=int(re1 and not conflict)
        output=(memory[ra0] if v0 else 0)|((memory[ra1] if v1 else 0)<<width)|(v0<<(2*width))|(v1<<(2*width+1))|(conflict<<(2*width+2))
        coverage['read0']+=v0;coverage['read1']+=v1;coverage['conflict']+=conflict
        coverage['read1_only']+=int(v1 and not v0);coverage['idle']+=int(not we and not re0 and not re1)
        coverage['read_write_collision']+=int(we and ((v0 and wa==ra0) or (v1 and wa==ra1)))
        if v0: read.add(ra0)
        if v1: read.add(ra1)
        if we: memory[wa]=wd;written.add(wa);coverage['write']+=1
    coverage.update(written_addresses=len(written),read_addresses=len(read))
    return rows,coverage


def check_regfile(p,ctx):
    if p.width<8 or not 16<=p.depth<=256 or p.depth&(p.depth-1) or p.banks&(p.banks-1) or p.depth<2*p.banks:
        raise InvalidInput('Register-file geometry requires width 8..64, depth 16..256 and power-of-two banks/depth')
    files=sources(p.files,ctx.cwd)
    if any('`include' in path.read_text() for path in files): raise InvalidInput('Inline source dependencies')
    sha=fingerprint(files,p.top);rows,coverage=regfile_trace(p.width,p.depth,p.banks,p.random_cycles,p.seed)
    delay=2 if p.timing_fixture else 0;stimuli=rows+[(0,0,None)]*delay
    inputs=p.width+3*(p.depth-1).bit_length()+3;outputs=2*p.width+3
    stimulus=''.join(f'{(rst<<inputs)|x:x}\n' for rst,x,_ in stimuli)
    (ctx.run.work/'regfile.hex').write_text(stimulus)
    tb=ctx.run.work/'regfile_tb.sv';binary=ctx.run.work/'regfile.vvp'
    tb.write_text(f'''module regfile_tb;
reg clk=0,rst;
reg [{inputs-1}:0] x;
wire [{outputs-1}:0] y;
reg [{inputs}:0] vectors[0:{len(stimuli)-1}];
{p.top} dut(.clk(clk),.rst(rst),.x(x),.y(y));
integer i;
initial begin
$readmemh("regfile.hex",vectors);
for(i=0;i<{len(stimuli)};i=i+1) begin
{{rst,x}}=vectors[i];#4;$display("IC_RF %0d %h",i,y);
#1;clk=1;#5;clk=0;
end
$finish;
end
endmodule
''')
    build=run_process(['iverilog','-g2012','-s','regfile_tb','-o',str(binary),*map(str,files),str(tb)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'compile.log',timeout_s=p.timeout_s)
    if build.exit_code or build.timed_out or re.search(r'warning:.*(port|width|dangling)',build.text(),re.I):
        return Result(ok=False,data=dict(log=ctx.run.handle('compile.log')),note='Register-file interface compilation failed.')
    run=run_process(['vvp',str(binary)],cwd=ctx.run.work,log_path=ctx.run.artifacts/'regfile.log',timeout_s=p.timeout_s)
    text=run.text(limit_bytes=len(stimuli)*(outputs//4+100))
    observed=re.findall(r'^IC_RF ([0-9]+) ([0-9a-fAxXzZ]+)\s*$',text,re.M)
    if run.exit_code or run.timed_out or len(observed)!=len(stimuli) or [int(r[0]) for r in observed]!=list(range(len(stimuli))) or re.search(r'\b(ERROR|FATAL)\b',text):
        return Result(ok=False,data=dict(log=ctx.run.handle('regfile.log'),timed_out=run.timed_out),note='Incomplete register-file simulation.')
    mismatch=None
    for row,(_,_,expected) in zip(observed[delay:],rows):
        if expected is not None and (any(c in row[1].lower() for c in 'xz') or int(row[1],16)!=expected):
            mismatch=dict(cycle=int(row[0]),expected=expected,observed=row[1]);break
    unchanged=sha==fingerprint(files,p.top)
    passed=mismatch is None and unchanged and all(coverage.values()) and coverage['written_addresses']==coverage['read_addresses']==p.depth
    return Result(ok=passed,data=dict(passed=passed,source_sha256=sha,source_unchanged=unchanged,coverage=coverage,first_mismatch=mismatch,
        cycles=len(stimuli),observation_delay_cycles=delay,latency_cycles=1 if passed else None,initiation_interval=1 if passed else None,transaction_unit='command_batch',
        stimulus_sha256=hashlib.sha256(stimulus.encode()).hexdigest(),log=ctx.run.handle('regfile.log')),
        note='Independent logical-memory model checks read-before-write, bank conflict priority, rejected-read zeros, reset clearing and registered output cycles. II counts command batches, not completed reads. Bounded simulation is not unbounded formal proof.')
