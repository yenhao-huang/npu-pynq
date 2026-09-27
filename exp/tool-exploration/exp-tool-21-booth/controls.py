"""Small formal positive/negative controls, separate from substantial benchmarks."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch
here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser();parser.add_argument('--container',default=None)
args=parser.parse_args()
out=here/'output';out.mkdir(exist_ok=True)
a=dispatch('synth_booth_multiplier',dict(width=8,architecture='native'))['data']
b=dispatch('synth_booth_multiplier',dict(width=8,architecture='radix4'))['data']
payload=dict(reference_files=[a['core_path']],candidate_files=[b['core_path']],top='dut',input_width=16,output_width=16,container=args.container,timeout_s=60)
positive=dispatch('yosys_equivalence',payload)
bad=out/'wrong-sign.sv'
bad.write_text(Path(b['core_path']).read_text().replace('{8{a[7]}}',"8'd0"))
negative=dispatch('yosys_equivalence',dict(payload,candidate_files=[str(bad.resolve())]))
(out/'formal-controls.json').write_text(json.dumps(dict(positive=positive,negative=negative),indent=2)+'\n')
print('positive',positive['ok'],positive['run_id'],'negative rejected',not negative['ok'],negative['run_id'])
