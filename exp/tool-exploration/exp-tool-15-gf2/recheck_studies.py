"""Reprove every distinct successful affine study source pair with source guards.

Preserve old study envelopes. This produces additional correctness evidence,
without changing physical measurements or claiming a new routed build.
"""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch
from ic_core.tools.exploration_common import fingerprint
from ic_core.tools.synth.acceptance import studies

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser()
parser.add_argument('--container')
args=parser.parse_args()
groups={}
for path in Path('docs/goals/0928-tool-exploration/evidence').glob('*.json'):
    for study in studies(json.loads(path.read_text())):
        for case in study['data']['cases']:
            proof=case.get('formal',{})
            data=proof.get('data',{})
            if proof.get('ok') and data.get('method')=='exact_affine_gf2':
                key=(data['reference_sha256'],data['candidate_sha256'])
                group=groups.setdefault(key,dict(run_id=proof['run_id'],historical_runs=[]))
                if proof['run_id'] not in group['historical_runs']:
                    group['historical_runs'].append(proof['run_id'])
rows=[]
out=here/'output';out.mkdir(exist_ok=True)
for hashes,group in groups.items():
    paths=list(Path('.ic/runs').glob('*/*-'+group['run_id']+'-debug'))
    if len(paths)!=1:raise RuntimeError('Missing unique historical run')
    params=json.loads((paths[0]/'meta.json').read_text())['inputs']
    for role,expected in zip(('reference','candidate'),hashes):
        if fingerprint([Path(p) for p in params[role+'_files']],params['top'])!=expected:
            raise RuntimeError('Historical source no longer matches')
    params['container']=args.container
    result=dispatch('gf2_equivalence',params)
    rows.append(dict(historical_runs=group['historical_runs'],source_sha256=list(hashes),result=result))
    (out/'source-guard-rechecks.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(group['historical_runs'],result['ok'],result['run_id'],flush=True)
    assert result['ok']
