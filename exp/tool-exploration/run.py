"""Reproducible, individually runnable paper-informed tool experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import shutil
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/ic'))
from ic_core import dispatch
from ic_core.tools.exploration_common import fingerprint, sources

HERE = Path(__file__).resolve().parent
FIX = HERE/'fixtures'
CATALOG = [
 ('01','rtlrewriter','optimization_rules','Retrieve area rules and their semantic preconditions.'),
 ('02','rtlrewriter','width_advice','A 16-bit result holding the sum of two 4-bit unsigned values needs only 5 bits.'),
 ('03','rtlrewriter','comb_check','Mux-before-add preserves all 8192 binary input combinations.'),
 ('04','aspen','ppa_measure','Measure whether arithmetic sharing reduces routed FPGA LUTs or delay.'),
 ('05','aspen','ppa_provenance','Only measurements made with identical conditions can be compared.'),
 ('06','aspen','ppa_compare','Report measured reductions and regressions without hiding either.'),
 ('07','aspen','ppa_pareto','Retain every measured nondominated trade-off.'),
 ('08','ppartl','ppa_reward','Score only correct, feasible candidates against the same baseline.'),
 ('09','rtlrewriter','ppa_select','Choose an affordable rewrite while retaining exploration of unvisited actions.'),
 ('10','rtlrewriter','rtl_evaluate','Verification precedes measurement and rejects an intentionally incorrect rewrite.'),
]


def check_payload(candidate='candidate.sv'):
 return dict(reference_files=['exp/tool-exploration/fixtures/baseline.sv'],
             candidate_files=['exp/tool-exploration/fixtures/'+candidate],top='dut',
             inputs=[{'name':n,'width':4} for n in ('a','b','c')]+[{'name':'sel','width':1}],
             outputs=[{'name':'y','width':4}],combinational_contract=True)


def invoke(name, payload):
 result=dispatch(name,payload,cwd=ROOT)
 if not result['ok']:
  raise RuntimeError(json.dumps(result))
 return result


def runtime_identity():
 import pydantic
 identity={'python':platform.python_version(),'pydantic':pydantic.__version__}
 for tool,flag in [('iverilog','-V'),('vivado','-version')]:
  executable=shutil.which(tool)
  identity[tool]={'executable':executable,'version':None}
  if executable:
   try:
    result=subprocess.run([executable,flag],capture_output=True,text=True,timeout=30,errors='replace')
    # Keep version/build lines only; never capture license messages or paths.
    lines=[line.strip() for line in (result.stdout+'\n'+result.stderr).splitlines()
           if re.search(r'Icarus Verilog version|Vivado v|SW Build|IP Build',line)]
    identity[tool]['version']=lines
   except (OSError,subprocess.TimeoutExpired):
    identity[tool]['version']='unavailable'
 return identity


def input_digest(environment):
 h=hashlib.sha256(json.dumps(environment,sort_keys=True).encode())
 for p in sorted((ROOT/'tools/ic/ic_core').rglob('*.py')) + sorted(HERE.glob('exp-tool-*/config.json')) + sorted(FIX.glob('*.sv')) + [Path(__file__)]:
  h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
 return h.hexdigest()


def run_one(op, config, dependencies):
 if op in ('optimization_rules','width_advice','ppa_select'):
  return invoke(op,config['payload'])
 if op=='comb_check':
  return invoke(op,check_payload())
 if op=='ppa_measure':
  rows=[]
  for name in ('baseline','candidate'):
   out=invoke(op,dict(files=[f'exp/tool-exploration/fixtures/{name}.sv'],top='dut',name=name,
                      constraint_ns=config['constraint_ns'],timeout_s=config['timeout_s']))
   rows.append(out)
  return {'ok':True,'measurements':rows}
 if op=='rtl_evaluate':
  positive=invoke(op,dict(check_payload(),measure_timeout_s=config['timeout_s']))
  negative=dispatch(op,check_payload('incorrect.sv'),cwd=ROOT)
  if negative['ok'] or negative['data'].get('stage')!='correctness':
   raise AssertionError('incorrect rewrite was not rejected before synthesis')
  return {'ok':True,'positive':positive,'negative_control':negative}
 rows=[r['data']['record'] for r in dependencies()['measurements']]
 if op=='ppa_pareto':
  return invoke(op,{'records':rows})
 payload={'baseline':rows[0],'candidate':rows[1]}
 if op=='ppa_reward':
  check=invoke('comb_check',check_payload())
  assert check['data']['reference_sha256']==rows[0]['source_sha256']
  assert check['data']['candidate_sha256']==rows[1]['source_sha256']
  payload.update(correctness_passed=True,weights=config['weights'],limits=config['limits'])
 return invoke(op,payload)


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--tool',default='all',choices=['all']+[r[0] for r in CATALOG])
 parser.add_argument('--output',type=Path,default=HERE/'output')
 parser.add_argument('--resume',action='store_true')
 args=parser.parse_args()
 args.output.mkdir(parents=True,exist_ok=True)
 environment=runtime_identity()
 digest=input_digest(environment)
 completed={}
 def execute(row):
  ident,paper,op,hypothesis=row
  if ident in completed: return completed[ident]
  folder=HERE/f'exp-tool-{ident}-{paper}'
  config=json.loads((folder/'config.json').read_text(encoding='utf-8-sig'))
  target=args.output/f'{ident}.json'
  if args.resume and target.exists():
   cached=json.loads(target.read_text())
   if cached['input_sha256']==digest and cached['result']['ok']:
    print(f'{ident} {op}: reused matching completed result',flush=True)
    completed[ident]=cached['result']
    return cached['result']
  print(f'{ident} {op}: running',flush=True)
  result=run_one(op,config,lambda:execute(CATALOG[3]))
  evidence={'tool_id':ident,'operation':op,'paper':paper,'hypothesis':hypothesis,
            'input_sha256':digest,'environment':environment,'utc':datetime.now(timezone.utc).isoformat(),
            'command':f'python exp/tool-exploration/run.py --tool {ident} --output {args.output.as_posix()}',
            'config':config,'result':result,
            'decision':'Inspect measured trade-offs and retain only correctness-checked candidates; no NPU or power claim.'}
  temp=target.with_suffix('.tmp'); temp.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8'); temp.replace(target)
  print(f'{ident} {op}: passed; {target}',flush=True)
  completed[ident]=result
  return result
 for row in CATALOG:
  if args.tool in ('all',row[0]): execute(row)
 return 0


if __name__=='__main__':
 raise SystemExit(main())
