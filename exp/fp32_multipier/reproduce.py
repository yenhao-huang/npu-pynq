"""Replay frozen RTL into fresh run directories and require exact PPA agreement."""
import argparse
import json
from run_eval import ROOT,evaluate
p=argparse.ArgumentParser();p.add_argument('names',nargs='+');p.add_argument('--prefix',default='reproduce');a=p.parse_args()
results=[]
for name in a.names:
    old=json.loads((ROOT/'runs'/name/'metrics.json').read_text())
    assert old['status']=='pass'
    fresh=evaluate(f'{a.prefix}_{name}',old['mode'],old['mul'],old['lzd'],old['target_delay_ps'],True,old.get('abc_fast',False),old.get('abc_constrained',False),ROOT/'runs'/name/'design.sv')
    assert fresh['status']=='pass',fresh
    assert fresh['checks']==fresh['gate_checks']==16777216
    assert fresh['rtl_sha256']==old['rtl_sha256']
    for field in ['area_um2','delay_ps','adp_um2_ps']:
        assert abs(fresh[field]-old[field])<=max(1e-6,abs(old[field])*1e-6),(field,fresh[field],old[field])
    results.append({'source':name,'reproduction':fresh['name'],'agreement':'within 1 ppm','rtl_sha256':fresh['rtl_sha256']})
(ROOT/f'{a.prefix}_verification.json').write_text(json.dumps(results,indent=2))
print('PASS frozen-source reproductions:',len(results))
