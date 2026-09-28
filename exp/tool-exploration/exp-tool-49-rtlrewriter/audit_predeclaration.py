"""Verify that study cases and objectives existed before physical child runs."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from ic_core.tools.synth.acceptance import studies


here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
evidence_root=Path('docs/goals/0928-tool-exploration/evidence')
runs_root=Path('.ic/runs')
seen={}

for evidence_path in sorted(evidence_root.glob('*.json')):
    for study in studies(json.loads(evidence_path.read_text())):
        run_id=study.get('run_id')
        if not run_id: raise RuntimeError('Study lacks a run identifier: '+str(evidence_path))
        canonical=json.dumps(study,sort_keys=True)
        previous=seen.setdefault(run_id,dict(study=study,canonical=canonical,evidence_files=[]))
        if previous['canonical']!=canonical: raise RuntimeError('Conflicting study envelope: '+run_id)
        previous['evidence_files'].append(str(evidence_path))

rows=[];measurement_runs=set();declared_cases=0
for run_id,item in sorted(seen.items()):
    matches=list(runs_root.glob('*/*-'+run_id+'-pipeline'))
    if len(matches)!=1: raise RuntimeError('Missing unique architecture sweep run: '+run_id)
    run=matches[0];meta=json.loads((run/'meta.json').read_text());out=json.loads((run/'out.json').read_text())
    study=item['study']
    if (meta.get('op')!='architecture_sweep' or meta.get('state') not in ('succeeded','failed')
            or out!=study or out.get('ok')!=(meta.get('state')=='succeeded') or meta.get('started_at') is None):
        raise RuntimeError('Study run envelope mismatch: '+run_id)
    started=datetime.fromisoformat(meta['started_at'])
    declarations={case['name']:case for case in meta['inputs']['cases']}
    if len(declarations)!=len(meta['inputs']['cases']): raise RuntimeError('Duplicate case name: '+run_id)
    checkpoint=Path(study['data']['checkpoint_directory'])
    if not checkpoint.is_dir(): raise RuntimeError('Missing checkpoint directory: '+run_id)
    children=[]
    for case in study['data']['cases']:
        declared=declarations.get(case['name'])
        if not declared or (case['generator'],case['objective'])!=(declared['generator'],declared['objective']):
            raise RuntimeError('Observed case differs from pre-run declaration: '+run_id+'/'+case['name'])
        for role in ('baseline','candidate'):
            envelope=json.loads((checkpoint/(case['name']+'-'+role+'-generate.json')).read_text())
            if envelope['operation']!=case['generator'] or envelope['payload']!=declared[role]:
                raise RuntimeError('Generation checkpoint differs from declaration: '+run_id+'/'+case['name'])
        for pair in case.get('records',[]):
            for record in pair:
                child_id=record['evidence']['metrics'].split('/',1)[0]
                child_matches=list(runs_root.glob('*/*-'+child_id+'-synth'))
                if len(child_matches)!=1: raise RuntimeError('Missing measurement child: '+child_id)
                child=json.loads((child_matches[0]/'meta.json').read_text())
                if child.get('op')!='clocked_ppa' or datetime.fromisoformat(child['started_at'])<started:
                    raise RuntimeError('Measurement predates declaration envelope: '+child_id)
                children.append(child_id);measurement_runs.add(child_id)
        declared_cases+=1
    rows.append(dict(run_id=run_id,started_at=meta['started_at'],study_name=meta['inputs']['study_name'],
                     declared_cases=len(declarations),measurement_runs=len(set(children)),
                     envelope_sha256=hashlib.sha256((run/'out.json').read_bytes()).hexdigest(),
                     evidence_files=sorted(set(item['evidence_files']))))

result=dict(scope='Ordering audit of architecture-sweep envelopes and their successful physical children.',
            studies=len(rows),declared_case_rows=declared_cases,
            unique_successful_measurement_runs=len(measurement_runs),rows=rows,passed=bool(rows),
            note='Each objective and baseline/candidate payload is present in the parent run input and generation checkpoints before every successful clocked child starts. This does not prove when an idea was conceived or authenticate deleted external history.')
output=here/'output';output.mkdir(exist_ok=True)
(output/'predeclaration-ordering.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['studies'],'studies and',result['unique_successful_measurement_runs'],'measurement runs passed:',result['passed'],flush=True)
assert result['passed']
