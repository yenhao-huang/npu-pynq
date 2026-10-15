"""Acceptance audit against actual snapshots, logs, and replay results."""
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/metrics.json')]
original=[r for r in rows if not r['name'].startswith(('reproduce_','audit_'))]
passed=[r for r in original if r['status']=='pass']
assert len(passed)>10
assert len({r['rtl_sha256'] for r in passed})>10
assert not any(r['status']=='running' for r in rows)
for r in passed:
    d=ROOT/'runs'/r['name']
    assert sha(d/'design.sv')==r['rtl_sha256']
    for field,log in [('checks','simulation.log'),('gate_checks','gate_simulation.log')]:
        assert int(re.search(r'PASS checks=(\d+)',(d/log).read_text()).group(1))==r[field]
    a=float(re.search(r'Chip area for module.*?:\s*([\d.]+)',(d/'synthesis.log').read_text()).group(1))
    t=float(re.search(r'^\s*([\d.]+)\s+data arrival time\s*$',(d/'sta.log').read_text(),re.M).group(1))
    assert abs(a-r['area_um2'])<1e-8 and abs(t-r['delay_ps'])<1e-8
    assert abs(a*t-r['adp_um2_ps'])<1e-6
    assert re.search(r'^Total\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+',(d/'sta.log').read_text(),re.M)
ranked=sorted(passed,key=lambda r:(r['adp_um2_ps'],r['name']))
expected=[];seen=set()
for r in ranked:
    if r['rtl_sha256'] not in seen:expected.append(r['name']);seen.add(r['rtl_sha256'])
    if len(expected)==3:break
top=json.loads((ROOT/'top3.json').read_text())
assert [r['name'] for r in top]==expected
replays=json.loads((ROOT/'reproduce_verification.json').read_text())
assert [r['source'] for r in replays]==expected
for replay in replays:
    old=next(r for r in rows if r['name']==replay['source'])
    new=next(r for r in rows if r['name']==replay['reproduction'])
    assert new['status']=='pass' and new['checks']==new['gate_checks']==4096**2
    assert sha(ROOT/'runs'/new['name']/'mapped.v')==new['mapped_sha256']
    assert old['rtl_sha256']==new['rtl_sha256']
    for field in ['area_um2','delay_ps','adp_um2_ps']:assert old[field]==new[field]
assert any(r['area_um2']<=79 and r['delay_ps']<=891 for r in passed)
for name in ['report.md','reproduce.md','run_eval.py','sta.py','environment.json']:
    assert (ROOT/name).stat().st_size>0
assert 'REPRODUCTION PENDING' not in (ROOT/'report.md').read_text(encoding='utf-8')
result={'status':'pass','successful_original_evaluations':len(passed),'distinct_passing_rtl':len({r['rtl_sha256'] for r in passed}),'failed_attempts':[r['name'] for r in original if r['status']=='failed'],'top3':expected,'full_domain_checks_per_replay_stage':4096**2,'exact_replay_agreement':True,'absolute_paper_reference_envelope_met':True,'paper_equivalent_fp16_reproduction_claimed':False}
(ROOT/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
