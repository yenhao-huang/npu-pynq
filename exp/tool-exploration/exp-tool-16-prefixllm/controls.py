"""Substantial arithmetic graph controls shared by operations 16-18."""
import json
import os
from pathlib import Path
from ic_core import dispatch

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
out=here/'output'/'controls';out.mkdir(parents=True,exist_ok=True)
rows=[]
cases=[
    ('prefix16','synth_prefix_adder',dict(width=16,architecture='native'),dict(width=16,architecture='kogge_stone'),' | (',' & ('),
    ('prefix32','synth_prefix_adder',dict(width=32,architecture='native'),dict(width=32,architecture='sklansky'),' | (',' & ('),
    ('csd16','synth_csd_multiplier',dict(width=16,constant=255,architecture='native'),dict(width=16,constant=255,architecture='csd'),' - ',' + '),
    ('csd32','synth_csd_multiplier',dict(width=32,constant=65535,architecture='native'),dict(width=32,constant=65535,architecture='csd'),' - ',' + '),
    ('mcm16','synth_mcm',dict(width=16,constants=[45,51,85,255],architecture='independent'),dict(width=16,constants=[45,51,85,255],architecture='shared'),' + ',' - '),
    ('mcm32','synth_mcm',dict(width=32,constants=[45,51,85,255],architecture='independent'),dict(width=32,constants=[45,51,85,255],architecture='shared'),' + ',' - '),
]
for name,op,baseline_params,candidate_params,target,replacement in cases:
    baseline=dispatch(op,baseline_params);candidate=dispatch(op,candidate_params)
    assert baseline['ok'] and candidate['ok']
    source=Path(candidate['data']['core_path']).read_text()
    assert target in source
    mutant=out/(name+'-mutant.sv');mutant.write_text(source.replace(target,replacement,1))
    for label,path,expected in [('positive',candidate['data']['core_path'],True),('mutation',str(mutant),False)]:
        result=dispatch('vector_equivalence',dict(reference_files=[baseline['data']['core_path']],candidate_files=[path],top='dut',input_width=baseline['data']['input_width'],output_width=baseline['data']['output_width'],random_vectors=4096,seed=928,combinational_contract=True))
        rows.append(dict(case=name,operation=op,label=label,expected=expected,
            baseline_sha256=baseline['data']['source_sha256'],candidate_sha256=candidate['data']['source_sha256'],
            mutation=None if expected else dict(target=target,replacement=replacement),result=result))
        (out/'controls.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(name,label,result['ok'],result['run_id'],flush=True)
        assert result['ok']==expected
        assert (result['data']['first_mismatch'] is None)==expected
