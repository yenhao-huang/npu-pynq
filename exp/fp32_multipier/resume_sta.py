"""Resume STA only from a completed, verified synthesis checkpoint."""
import json
import sys
import time
from run_eval import ROOT,IMAGE,run,digest
from sta import measure
for name in sys.argv[1:]:
    dest=ROOT/'runs'/name;p=dest/'metrics.json';m=json.loads(p.read_text())
    assert m['status']!='running','Do not resume a live run'
    assert m.get('checks',0)>0 and m.get('gate_checks',0)>0
    assert digest(dest/'design.sv')==m['rtl_sha256']
    assert digest(ROOT/'build/pdk/asap7_rvt_tt.lib')==m['lib_sha256']
    m.setdefault('prior_errors',[]).append(m.pop('error','STA adapter correction'))
    if (dest/'sta.log').exists():
        import shutil
        shutil.copy2(dest/'sta.log',dest/f'sta_previous_{int(time.time())}.log')
    try:
        m.update(measure(dest,run,IMAGE));m['adp_um2_ps']=m['area_um2']*m['delay_ps'];m['status']='pass';m['image']=IMAGE
    except Exception as e:m['status']='failed';m['error']=str(e)
    m['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');p.write_text(json.dumps(m,indent=2));print(json.dumps(m),flush=True)
