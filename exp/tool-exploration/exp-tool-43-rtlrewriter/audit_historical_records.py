"""Authenticate pre-coverage clocked records against their preserved local runs.

This checks source bytes, run envelopes, parsed metrics and raw report digests.
It cannot reconstruct clock coverage or check_timing reports that were not
emitted by the historical in-memory Vivado flow.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from ic_core.tools.exploration_common import fingerprint
from ic_core.tools.synth.acceptance import studies
from ic_core.tools.synth.clocked import utilization


here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
evidence_root=Path('docs/goals/0928-tool-exploration/evidence')
runs_root=Path('.ic/runs')
records={}
sources=[]

for evidence_path in sorted(evidence_root.glob('*.json')):
    document=json.loads(evidence_path.read_text())
    for study in studies(document):
        for case in study['data']['cases']:
            for pair in case.get('records',[]):
                for record in pair:
                    handles=record.get('evidence',{})
                    if 'coverage' in handles or 'constraint_checks' in handles: continue
                    run_id=handles.get('metrics','').split('/',1)[0]
                    if not run_id: raise RuntimeError('Historical record lacks a run handle')
                    canonical=json.dumps(record,sort_keys=True)
                    previous=records.setdefault(run_id,dict(record=record,canonical=canonical,evidence_files=[]))
                    if previous['canonical']!=canonical: raise RuntimeError('Conflicting record for '+run_id)
                    previous['evidence_files'].append(str(evidence_path))

rows=[]
for run_id,item in sorted(records.items()):
    matches=list(runs_root.glob('*/*-'+run_id+'-synth'))
    if len(matches)!=1: raise RuntimeError('Missing unique preserved run: '+run_id)
    run=matches[0];meta=json.loads((run/'meta.json').read_text());out=json.loads((run/'out.json').read_text())
    record=item['record']
    if (meta.get('run_id')!=run_id or meta.get('op')!='clocked_ppa'
            or meta.get('state')!='succeeded' or not out.get('ok')
            or out.get('run_id')!=run_id or out.get('data',{}).get('record')!=record):
        raise RuntimeError('Run envelope or exported record mismatch: '+run_id)
    expected_handles={key:f'{run_id}/{name}' for key,name in (
        ('metrics','clocked.txt'),('utilization','utilization.txt'),
        ('timing','timing.txt'),('log','clocked.log'))}
    if record.get('evidence')!=expected_handles:
        raise RuntimeError('Historical evidence handles do not identify their run: '+run_id)
    params=meta['inputs'];files=[]
    for descriptor in params['files']:
        path=Path(descriptor['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=descriptor['sha256']:
            raise RuntimeError('Measured source file changed: '+run_id)
        files.append(path)
    if fingerprint(files,params['top'])!=record['source_sha256']:
        raise RuntimeError('Measured source fingerprint mismatch: '+run_id)
    artifacts=run/'artifacts';digests={}
    declared={row['name']:row['bytes'] for row in meta['artifacts']}
    for name in ('clocked.txt','utilization.txt','timing.txt','clocked.log'):
        path=artifacts/name
        if not path.is_file() or declared.get(name)!=path.stat().st_size:
            raise RuntimeError('Missing or size-mismatched raw report: '+run_id+'/'+name)
        digests[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=dict(line.split('=',1) for line in (artifacts/'clocked.txt').read_text().splitlines() if '=' in line)
    resources=utilization((artifacts/'utilization.txt').read_text())
    slack=float(raw['slack_ns']);period=float(params['period_ns']);critical=period-slack
    calculated=dict(resources,slack_ns=slack,critical_period_ns=critical,
                    estimated_fmax_mhz=1000/critical,
                    throughput_mtransactions_s=1000/critical/params['initiation_interval'])
    for name,value in calculated.items():
        if name not in record['metrics'] or not math.isclose(record['metrics'][name],value,rel_tol=1e-9,abs_tol=1e-9):
            raise RuntimeError('Parsed metric mismatch: '+run_id+'/'+name)
    if raw['version']!=record['version'] or raw['build']!=record['build']:
        raise RuntimeError('Vivado identity mismatch: '+run_id)
    tcl=run/'work'/'clocked.tcl';script=tcl.read_text()
    required=(f'create_project -in_memory -part {record["part"]}',
              f'synth_design -mode out_of_context -top {params["top"]}',
              f'create_clock -name ic_clock -period {period}',
              f'place_design -directive {record["directive"]}',
              f'route_design -directive {record["directive"]}',
              'get_timing_paths -delay_type max -from [all_registers -clock ic_clock] -to [all_registers -clock ic_clock]',
              'report_utilization -file','report_timing -of_objects $paths -file')
    if not all(fragment in script for fragment in required):
        raise RuntimeError('Historical Tcl contract mismatch: '+run_id)
    sources.extend(str(path) for path in files)
    rows.append(dict(run_id=run_id,name=record['name'],source_sha256=record['source_sha256'],
                     raw_artifact_sha256=digests,tcl_sha256=hashlib.sha256(tcl.read_bytes()).hexdigest(),
                     evidence_files=sorted(set(item['evidence_files'])),metrics=record['metrics']))

result=dict(scope='Authenticity audit of preserved pre-coverage clocked runs; not a timing-coverage audit.',
            historical_records=len(rows),unique_source_files=len(set(sources)),rows=rows,
            passed=bool(rows),limitations=[
                'Historical runs did not emit register/clock coverage or check_timing reports.',
                'In-memory projects have no saved checkpoint, so missing coverage cannot be reconstructed.',
                'Latency and initiation interval remain independently checked caller inputs.'])
output=here/'output';output.mkdir(exist_ok=True)
(output/'historical-record-authenticity.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['historical_records'],'historical records passed:',result['passed'],flush=True)
assert result['passed']
