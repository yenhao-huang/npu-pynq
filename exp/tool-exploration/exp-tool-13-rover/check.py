"""Formal regression with positive, incorrect and interface-mismatch controls."""
import argparse
import json
from pathlib import Path
from ic_core import dispatch

parser=argparse.ArgumentParser()
parser.add_argument('--container')
parser.add_argument('--controls-only', action='store_true')
args=parser.parse_args()
out=Path(__file__).resolve().parent/'output'
out.mkdir(exist_ok=True)
ref=dispatch('synth_adder_tree',dict(width=16,lanes=4,architecture='serial'))['data']
candidate=dispatch('synth_adder_tree',dict(width=16,lanes=4,architecture='balanced'))['data']
wrong=out/'incorrect.sv'
wrong.write_text('module dut(input [63:0] x, output [15:0] y); assign y=x[15:0]; endmodule')
for label,path,width in [('positive',candidate['core_path'],64),('negative',str(wrong),64),('wrong_width',candidate['core_path'],32)]:
    result=dispatch('yosys_equivalence',dict(reference_files=[ref['core_path']],candidate_files=[path],top='dut',input_width=width,output_width=16,container=args.container,timeout_s=120))
    (out/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(label,result['ok'],result['run_id'],flush=True)
    assert result['ok']==(label=='positive')
if args.controls_only:
    raise SystemExit(0)
for width,lanes in [(16,4),(16,16),(32,16)]:
    ref=dispatch('synth_adder_tree',dict(width=width,lanes=lanes,architecture='serial'))['data']
    for architecture in ['balanced','compressor']:
        candidate=dispatch('synth_adder_tree',dict(width=width,lanes=lanes,architecture=architecture))['data']
        result=dispatch('yosys_equivalence',dict(reference_files=[ref['core_path']],candidate_files=[candidate['core_path']],top='dut',input_width=width*lanes,output_width=width,container=args.container,timeout_s=120))
        label=f'w{width}-n{lanes}-{architecture}'
        (out/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(label,result['ok'],result['run_id'],flush=True)
