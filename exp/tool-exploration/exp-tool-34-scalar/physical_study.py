"""Registered FIFO physical study, gated by core and delayed-fixture scoreboards."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch
here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser();parser.add_argument('--distributed-revision',action='store_true');args=parser.parse_args()
payload=json.loads((here/('physical-distributed-config.json' if args.distributed_revision else 'physical-config.json')).read_text())
payload['formal_container']='codex-sandbox-agent-workspace'
result=dispatch('architecture_sweep',payload)
out=here/'output';out.mkdir(exist_ok=True)
(out/('physical-'+result['run_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(result['ok'],result['run_id'],[(c['name'],c['status']) for c in result['data']['cases']],flush=True)
