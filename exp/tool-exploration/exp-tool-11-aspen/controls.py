"""Audit one substantial routed record and reject a source with no clock port."""
import json
import os
from pathlib import Path
from ic_core import dispatch

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
out=here/'output';out.mkdir(exist_ok=True)
study=json.loads(Path('docs/goals/0928-tool-exploration/evidence/dot-prefix-physical-study.json').read_text())
case=study['data']['cases'][0]
generated=json.loads((Path(study['data']['checkpoint_directory'])/(case['name']+'-baseline-generate.json')).read_text())['result']['data']
record=case['records'][0][0]
audit=dispatch('timing_constraint_audit',dict(record=record,files=[generated['core_path'],generated['wrapper_path']],top='registered_dut'))
assert audit['ok'] and audit['data']['coverage']['clock_count']==1
bad=out/'missing-clock.sv';bad.write_text('module dut(input [7:0] x,output reg [7:0] y); always @* y=x; endmodule\n')
negative=dispatch('clocked_ppa',dict(files=[str(bad)],top='dut',clock_port='clk',name='missing-clock-control',period_ns=5,latency_cycles=1,initiation_interval=1,timeout_s=180))
assert not negative['ok'] and not negative['data']['timed_out']
result=dict(scope='Operation 11 control using one 16-bit, eight-lane dot-product routed record plus an actual missing-clock rejection.',positive_audit=audit,negative_missing_clock=negative)
(out/'controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(audit['run_id'],negative['run_id'],flush=True)
