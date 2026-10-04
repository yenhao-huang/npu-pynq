"""Audit original RTL hashes, simulation evidence, and measured PPA."""
import hashlib
import json
import re
from pathlib import Path
from summarize import ranked
ROOT=Path(__file__).resolve().parent
valid,distinct=ranked()
assert valid,'No passing evaluations'
for r in valid:
    d=ROOT/'runs'/r['name']
    assert hashlib.sha256((d/'design.sv').read_bytes()).hexdigest()==r['rtl_sha256'],r['name']
    assert re.search(r'module\s+fpadd_fp16\b',(d/'design.sv').read_text(encoding='utf-8-sig'))
    for log,key in [('simulation.log','checks'),('gate_simulation.log','gate_checks')]:
        text=(d/log).read_text()
        assert f"PASS checks={r[key]}" in text
        assert r[key]==(4294967296 if r['exhaustive'] else 3359296)
    timing=(d/'sta.log').read_text()
    delay=float(re.search(r'^\s*([\d.]+)\s+data arrival time\s*$',timing,re.M).group(1))
    area=float(re.search(r'Chip area for module.*?:\s*([\d.]+)',(d/'synthesis.log').read_text()).group(1))
    assert area==r['area_um2'] and delay==r['delay_ps']
    assert area*delay==r['adp_um2_ps']
    assert r['meets_mapping_target']==(delay<=r['target_delay_ps'])
print(f'PASS evidence audit: {len(valid)} evaluations, {len(distinct)} distinct RTL')

# Publication acceptance additionally requires complete, fresh top-three evidence.
import sys
if '--require-complete' in sys.argv:
    assert len(valid)>10
    assert not any(json.loads(p.read_text())['status']=='running' for p in (ROOT/'runs').glob('[0-9][0-9][0-9]_*/metrics.json'))
    top=json.loads((ROOT/'top3.json').read_text())
    assert [r['name'] for r in top]==[r['name'] for r in distinct[:3]]
    assert len(top)==3 and len({r['rtl_sha256'] for r in top})==3
    replays=json.loads((ROOT/'replay_verification.json').read_text())
    assert {r['original'] for r in replays}=={r['name'] for r in top}
    for record in replays:
        original=next(r for r in top if r['name']==record['original'])
        directory=ROOT/'runs'/record['replay']
        replay=json.loads((directory/'metrics.json').read_text())
        assert replay['status']=='pass'
        assert hashlib.sha256((directory/'design.sv').read_bytes()).hexdigest()==original['rtl_sha256']
        for key in ['area_um2','delay_ps','adp_um2_ps','lib_sha256','rtl_sha256','checks','gate_checks']:
            assert replay[key]==original[key],key
        for log,key in [('simulation.log','checks'),('gate_simulation.log','gate_checks')]:
            assert f"PASS checks={replay[key]}" in (directory/log).read_text()
    print('PASS completion gate: >10 evaluations, distinct top three, fresh verified replay')
