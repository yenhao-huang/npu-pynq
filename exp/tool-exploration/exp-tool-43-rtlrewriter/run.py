"""Real routed coverage checks, including an unclocked sequential domain."""
import json
import os
from pathlib import Path
from ic_core import dispatch
here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
config=json.loads((here/'config.json').read_text())
out=here/'output';out.mkdir(exist_ok=True)
results=[]
for width in config['widths']:
    d=dispatch('synth_prefix_adder',dict(width=width,architecture='native'))['data']
    files=[d['core_path'],d['wrapper_path']]
    measurement=dispatch('clocked_ppa',dict(files=files,top='registered_dut',name=f'audit{width}',period_ns=config['period_ns'],optimization_mode=config['optimization_mode'],latency_cycles=1,initiation_interval=1))
    audit=dispatch('timing_constraint_audit',dict(record=measurement['data']['record'],files=files,top='registered_dut')) if measurement['ok'] else None
    results.append(dict(name=f'w{width}',measurement=measurement,audit=audit))
    (out/'audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print(width,measurement['ok'],None if audit is None else audit['ok'],flush=True)
source=out/'unclocked.sv'
source.write_text("""module unclocked(input clk,input aux,input [31:0] x,output reg [31:0] y);
reg [31:0] q;
reg extra;
always @(posedge aux) extra<=x[0];
always @(posedge clk) begin q<=x;y<=q+extra;end
endmodule
""")
measurement=dispatch('clocked_ppa',dict(files=[str(source.resolve())],top='unclocked',name='unclocked-domain-negative',period_ns=config['period_ns'],optimization_mode=config['optimization_mode'],latency_cycles=1,initiation_interval=1))
audit=dispatch('timing_constraint_audit',dict(record=measurement['data']['record'],files=[str(source.resolve())],top='unclocked')) if measurement['ok'] else None
results.append(dict(name='unclocked_negative',measurement=measurement,audit=audit))
(out/'audit.json').write_text(json.dumps(results,indent=2)+'\n')
print('negative',measurement['ok'],None if audit is None else audit['ok'],flush=True)
