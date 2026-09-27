"""Real SAT controls for complete bit proofs and conservative product abstraction."""
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
out=here/'output'/'bitwise';out.mkdir(parents=True,exist_ok=True)
rows=[]
header='module dut(input [23:0] x,output [15:0] y); wire [15:0] p=x[7:0]*x[15:8]; '
for mode in ['bitwise','bitwise_products']:
    for label,expression,expected in [
        ('positive',"{8'b0,x[23:16]}+p",True),
        ('high_bit_mutation',"({8'b0,x[23:16]}+p)^16'h8000",False),
        ('product_mutation',"{8'b0,x[23:16]}+(x[7:0]*x[23:16])",False),
        ('undefined_product',"x[7:0]*8'bx",False),
        ('partial_function',"x[7:0]*(x[15:8]/x[23:16])",False),
    ]:
        ref=out/'reference.sv';cand=out/'candidate.sv'
        reference=header+"assign y=p+{8'b0,x[23:16]}; endmodule"
        candidate=header+'assign y='+expression+'; endmodule'
        # A self-equivalence proof must also reject an undefined output.
        if label in ('undefined_product','partial_function'): reference=candidate
        ref.write_text(reference);cand.write_text(candidate)
        result=dispatch('yosys_equivalence',dict(reference_files=[str(ref)],candidate_files=[str(cand)],top='dut',input_width=24,output_width=16,normalization=mode,container=args.container,timeout_s=30))
        rows.append(dict(mode=mode,label=label,expected=expected,result=result))
        (out/'controls.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(mode,label,result['ok'],result['run_id'],flush=True)
        assert result['ok']==expected
