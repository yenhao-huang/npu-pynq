"""Replay selected frozen RTL and assert exact functional/PPA agreement."""
import json
import sys
from datetime import datetime
from pathlib import Path
from run_eval import evaluate,digest
ROOT=Path(__file__).resolve().parent

def main():
    selected=json.loads((ROOT/'top3.json').read_text())
    names=sys.argv[1:]
    if names: selected=[r for r in selected if r['name'] in names]
    if not selected: raise RuntimeError('No selected top-three designs')
    report=[]
    for old in selected:
        source=ROOT/'runs'/old['name']/'design.sv'
        assert digest(source)==old['rtl_sha256']
        name='replay_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+old['name']
        new=evaluate(name,old['mode'],old['mul'],old['lzd'],old['target_delay_ps'],constrained=old['abc_constrained'],source=source)
        assert new['status']=='pass',new
        for key in ['rtl_sha256','lib_sha256','checks','gate_checks','area_um2','delay_ps','adp_um2_ps']:
            assert new[key]==old[key],(key,old[key],new[key])
        report.append({'original':old['name'],'replay':name,'agreement':'exact','checks':new['checks'],'gate_checks':new['gate_checks']})
        (ROOT/'replay_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS fresh replay agreement')
if __name__=='__main__': main()
