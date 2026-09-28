"""Substantial network controls shared by operations 23-28."""
import json
import os
from pathlib import Path
from ic_core import dispatch

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
out=here/'output'/'controls';out.mkdir(parents=True,exist_ok=True)
rows=[]
cases=[
    ('popcount','synth_popcount',{},' + ',' - '),
    ('priority','synth_priority_encoder',{},' | ',' & '),
    ('leading_zero','synth_leading_zero',{},'~|x[','|x['),
    ('barrel','synth_barrel_shifter',{},' << ',' >> '),
    ('onehot','synth_onehot_mux',dict(lanes=16),' | ',' ^ '),
    ('argmax','synth_argmax_tree',dict(lanes=16),' >= ',' > '),
]
for short,op,extra,target,replacement in cases:
    for width in ((64,128) if short in ('popcount','priority','leading_zero') else (16,32)):
        baseline=dispatch(op,dict(width=width,architecture='linear',**extra))
        candidate=dispatch(op,dict(width=width,architecture='tree',**extra))
        assert baseline['ok'] and candidate['ok']
        source=Path(candidate['data']['core_path']).read_text();assert target in source
        mutant=out/f'{short}-{width}-mutant.sv';mutant.write_text(source.replace(target,replacement,1))
        for label,path,expected in [('positive',candidate['data']['core_path'],True),('mutation',str(mutant),False)]:
            result=dispatch('vector_equivalence',dict(reference_files=[baseline['data']['core_path']],candidate_files=[path],top='dut',input_width=baseline['data']['input_width'],output_width=baseline['data']['output_width'],random_vectors=2048,seed=928,combinational_contract=True))
            rows.append(dict(operation=op,width=width,parameters=extra,label=label,expected=expected,
                mutation=None if expected else dict(target=target,replacement=replacement),result=result))
            (out/'controls.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(short,width,label,result['ok'],result['run_id'],flush=True)
            assert result['ok']==expected
            assert (result['data']['first_mismatch'] is None)==expected
