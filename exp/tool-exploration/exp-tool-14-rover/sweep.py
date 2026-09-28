"""Proof-gated retry of all four original reduction comparisons."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser()
parser.add_argument('--container')
parser.add_argument('--verify-only',action='store_true')
args=parser.parse_args()
payload=json.loads((here/'config-macc.json').read_text())
payload.update(formal_container=args.container,verify_only=args.verify_only)
result=dispatch('architecture_sweep',payload)
out=here/'output';out.mkdir(exist_ok=True)
(out/('study-'+result['run_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(result['ok'],result['run_id'],[(case['name'],case['status']) for case in result['data']['cases']],flush=True)
