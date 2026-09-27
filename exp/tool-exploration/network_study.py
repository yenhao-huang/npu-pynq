"""Run predeclared network experiments through the registry pipeline."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch

root=Path(__file__).resolve().parents[2]
here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--tool',choices=['16','17','18','23','24','25','26','27','28','all'],default='all')
parser.add_argument('--container')
parser.add_argument('--configuration',default='config.json',choices=['config.json','config-binary-search.json'])
parser.add_argument('--verify-only',action='store_true')
args=parser.parse_args()
os.chdir(root)
ids=['16','17','18','23','24','25','26','27','28'] if args.tool=='all' else [args.tool]
for id in ids:
    folder=next(here.glob('exp-tool-'+id+'-*'))
    payload=json.loads((folder/args.configuration).read_text())
    payload.update(formal_container=args.container,verify_only=args.verify_only)
    result=dispatch('architecture_sweep',payload)
    output=folder/'output'
    output.mkdir(exist_ok=True)
    (output/(result['run_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(id,result['ok'],result['run_id'],[(c['name'],c['status']) for c in result['data']['cases']],flush=True)
