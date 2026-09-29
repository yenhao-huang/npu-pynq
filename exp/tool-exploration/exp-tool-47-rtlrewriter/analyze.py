"""Retrospective source-bound FIFO storage ablations, preserving original objectives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from ic_core import dispatch
from ic_core.tools.exploration_common import fingerprint
from ic_core.tools.synth.fifo import FifoIn

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser()
parser.add_argument('--supplement',type=Path,help='Completed depth128 study JSON; adds full storage-by-capacity analysis.')
args=parser.parse_args()
evidence=Path('docs/goals/0928-tool-exploration/evidence')
inputs={}


def load(path):
    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=json.loads(path.read_text())
    if not result['ok']: raise ValueError('Study incomplete: '+str(path))
    return result['data']


def cell(study,case,role,levels):
    if case['status']!='measured': raise ValueError('Unmeasured cell')
    index=0 if role=='baseline' else 1
    checkpoint=Path(study['checkpoint_directory'])/(case['name']+'-'+role+'-generate.json')
    inputs[str(checkpoint)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    envelope=json.loads(checkpoint.read_text())
    if envelope['operation']!='synth_fifo' or not envelope['result']['ok']: raise ValueError('Wrong generator checkpoint')
    params=FifoIn.model_validate(envelope['payload'])
    design=envelope['result']['data']
    for offset,files,top in [(0,design['files'],design['top']),(1,design['timing_files'],design['timing_top'])]:
        check=case['checks'][index*2+offset]
        if not check['ok'] or not check['data']['passed'] or not check['data']['source_unchanged']:
            raise ValueError('Missing source-bound correctness')
        if fingerprint([Path(f) for f in files],top)!=check['data']['source_sha256']:
            raise ValueError('Correctness source differs from current files')
    return dict(levels=levels,configuration={k:getattr(params,k) for k in ('width','depth','architecture','memory_style')},
                status='measured',files=design['timing_files'],top=design['timing_top'],records=[pair[index] for pair in case['records']])


auto=load(evidence/'fifo-auto-physical-study.json')
distributed=load(evidence/'fifo-distributed-physical-study.json')
outputs=[]
for i in range(2):
    cells=[cell(auto,auto['cases'][i],'candidate',dict(memory_style=0)),
           cell(distributed,distributed['cases'][i],'candidate',dict(memory_style=1))]
    result=dispatch('candidate_ablation',dict(factors=dict(memory_style=['auto','distributed']),cells=cells))
    outputs.append(dict(configuration=cells[0]['configuration'],analysis=result))
if args.supplement:
    supplemental=load(args.supplement)
    cells=[cell(auto,auto['cases'][0],'candidate',dict(memory_style=0,depth=0)),
           cell(distributed,distributed['cases'][0],'candidate',dict(memory_style=1,depth=0)),
           cell(supplemental,supplemental['cases'][0],'baseline',dict(memory_style=0,depth=1)),
           cell(supplemental,supplemental['cases'][0],'candidate',dict(memory_style=1,depth=1))]
    result=dispatch('candidate_ablation',dict(factors=dict(memory_style=['auto','distributed'],depth=[64,128]),cells=cells))
    outputs.append(dict(configuration=dict(width=16,architecture='circular'),analysis=result))
out=here/'output';out.mkdir(exist_ok=True)
report=dict(input_sha256=inputs,analyses=outputs,retrospective=True,
    note='Storage effects are separate from shift-to-circular gains. Depth is a workload factor, not an optimization; cross-depth differences cannot qualify a win. Original study objectives and failures remain unchanged. Historical timing records lack the later coverage audit.')
path=out/('factorial-ablation.json' if args.supplement else 'storage-ablation.json')
path.write_text(json.dumps(report,indent=2)+'\n')
for row in outputs:
    data=row['analysis']['data']
    print(row['analysis']['run_id'],{m:[c['mean'] for c in data['effects'][m]['contrasts']] for m in ('luts','brams','throughput_mtransactions_s')},flush=True)
