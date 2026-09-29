"""Check substantial reductions and detect arithmetic/carry-path mutations."""
import json
import os
from pathlib import Path
from ic_core import dispatch

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
out=here/'output'/'controls';out.mkdir(parents=True,exist_ok=True)
rows=[]
for width in (16,32):
    baseline=dispatch('synth_adder_tree',dict(width=width,lanes=16,architecture='serial'))
    assert baseline['ok']
    for architecture in ('balanced','compressor'):
        generated=dispatch('synth_adder_tree',dict(width=width,lanes=16,architecture=architecture))
        assert generated['ok']
        source=Path(generated['data']['core_path']).read_text()
        target,replacement=(' + ',' - ') if architecture=='balanced' else ('<< 1','<< 2')
        assert target in source
        mutant=out/f'{architecture}-{width}-mutant.sv'
        mutant.write_text(source.replace(target,replacement,1))
        for label,candidate,expected in [('positive',generated['data']['core_path'],True),('mutation',str(mutant),False)]:
            result=dispatch('vector_equivalence',dict(reference_files=[baseline['data']['core_path']],candidate_files=[candidate],top='dut',input_width=width*16,output_width=width,random_vectors=4096,seed=928,combinational_contract=True))
            rows.append(dict(width=width,lanes=16,architecture=architecture,label=label,
                             mutation=None if expected else dict(target=target,replacement=replacement),
                             baseline=baseline,candidate=generated,result=result))
            (out/'controls.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(width,architecture,label,result['ok'],result['run_id'],flush=True)
            assert result['ok']==expected
            assert (result['data']['first_mismatch'] is None)==expected
