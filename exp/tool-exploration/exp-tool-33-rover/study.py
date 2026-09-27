"""Saturating add/subtract: source-bound simulation/SAT gates and paired physical runs."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch
here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser()
parser.add_argument('--verify-only',action='store_true')
parser.add_argument('--container',default=None)
args=parser.parse_args()
payload=json.loads((here/'config.json').read_text())
payload.update(verify_only=args.verify_only,formal_container=args.container)
result=dispatch('architecture_sweep',payload)
out=here/'output';out.mkdir(exist_ok=True)
(out/('study-'+result['run_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(result['ok'],result['run_id'],[(case['name'],case['status']) for case in result['data']['cases']],flush=True)
