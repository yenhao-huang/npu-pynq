"""Reject partial source semantics even when optimization removes the operation."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser()
parser.add_argument('--container')
args=parser.parse_args()
out=here/'output'/'sources';out.mkdir(parents=True,exist_ok=True)
plain='module dut(input [7:0] x,output [7:0] y); assign y=0; endmodule'
cases=[
    ('constant_division',"x/8'd1",True),
    ('self_division','x/x',False),
    ('masked_unknown',"8'b0*(x*8'bx)",False),
    ('cancelled_select','x[x]^x[x]',False),
    ('cancelled_division','(x/x)^(x/x)',False),
    ('cancelled_unknown',"(x*8'bx)^(x*8'bx)",False),
]
rows=[]
for label,expression,expected in cases:
    source='module dut(input [7:0] x,output [7:0] y); assign y='+expression+'; endmodule'
    for role in (['self'] if label!='cancelled_select' else ['self','reference','candidate']):
        ref,cand=out/'reference.sv',out/'candidate.sv'
        ref.write_text(plain if role=='candidate' else source)
        cand.write_text(plain if role=='reference' else source)
        result=dispatch('gf2_equivalence',dict(reference_files=[str(ref)],candidate_files=[str(cand)],input_width=8,output_width=8,container=args.container))
        rows.append(dict(label=label,role=role,expected=expected,result=result))
        (out/'controls.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(label,role,result['ok'],result.get('run_id'),flush=True)
        assert result['ok']==expected
        guards=result['data']['source_guards']
        assert all(not guard['data']['timed_out'] for guard in guards)
        assert all(guard['data']['interface_complete'] and guard['data']['source_unchanged'] for guard in guards)
        assert guards[-1]['data']['abstraction_defined']==expected
