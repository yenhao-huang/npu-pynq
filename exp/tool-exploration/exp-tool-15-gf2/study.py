"""Actual Yosys affine proof controls, including an Icarus-replayed witness."""
import json
import os
from pathlib import Path
import sys
from ic_core import dispatch
from ic_core.errors import InvalidInput
here=Path(__file__).resolve().parent;root=here.parents[2];os.chdir(root)
sys.path.insert(0,str(root/'tools/ic/tests'))
from test_arithmetic_exploration import simulate
out=here/'output';out.mkdir(exist_ok=True)
a=dispatch('synth_crc_parallel',dict(data_width=64,architecture='unrolled'))['data']
b=dispatch('synth_crc_parallel',dict(data_width=64,architecture='shared'))['data']
c=dispatch('synth_crc_parallel',dict(data_width=64,architecture='shared',polynomial=0x1EDC6F41))['data']
results={}
for name,candidate in [('positive',b),('negative',c)]:
    result=dispatch('gf2_equivalence',dict(reference_files=[a['core_path']],candidate_files=[candidate['core_path']],input_width=96,output_width=32,container='codex-sandbox-agent-workspace'))
    results[name]=result
    assert result['ok']==(name=='positive')
    if name=='negative':
        witness=result['data']['counterexample'];value=int(witness['x'],16)
        for label,design,key in [('reference',a,'reference_y'),('candidate',candidate,'candidate_y')]:
            stage=out/(result['run_id']+'-'+label);stage.mkdir(exist_ok=True)
            simulate(stage,design,[value],[int(witness[key],16)])
        results['witness_replayed_in_icarus']=True
    print(name,result['run_id'],result['ok'],flush=True)
nonlinear=out/'nonlinear.sv';nonlinear.write_text('module dut(input [1:0] x, output y); assign y=x[0]&x[1]; endmodule\n')
try:
    dispatch('gf2_equivalence',dict(reference_files=[str(nonlinear)],candidate_files=[str(nonlinear)],input_width=2,output_width=1,container='codex-sandbox-agent-workspace'))
except InvalidInput as error:
    results['nonlinear_rejected']=str(error)
else: raise AssertionError('nonlinear network accepted')
(out/'controls.json').write_text(json.dumps(results,indent=2)+'\n')
print('all affine controls completed',flush=True)
