"""Exercise the original ten tools in the predeclared 16-bit CSD study."""
import argparse
import copy
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.registry import find_op
from ic_core.tools.exploration_common import fingerprint
from run import CATALOG, runtime_identity

here=Path(__file__).resolve().parent;root=here.parents[1];os.chdir(root)
parser=argparse.ArgumentParser()
parser.add_argument('--tool',choices=['all']+[r[0] for r in CATALOG],default='all')
args=parser.parse_args()
config_path=here/'exp-tool-10-rtlrewriter/wide-workflow.json'
config=json.loads(config_path.read_text());environment=runtime_identity()
digest=hashlib.sha256(config_path.read_bytes()+Path(__file__).read_bytes()+json.dumps(environment,sort_keys=True).encode())
for op in [r[2] for r in CATALOG]+[config['generator']]:
 category,entry=find_op(op);backend=entry.In.model_fields['backend'].default or category.default_backend
 digest.update(Path(inspect.getsourcefile(getattr(category.backends[backend].impl,op))).read_bytes())
key=digest.hexdigest();folder=here/'exp-tool-10-rtlrewriter/output'/('wide-'+key[:16]);folder.mkdir(parents=True,exist_ok=True)
generated=folder/'generated.json'
if generated.exists(): designs=json.loads(generated.read_text())
else:
 designs=[dispatch(config['generator'],config[role]) for role in ('baseline','candidate')]
 generated.write_text(json.dumps(designs,indent=2)+'\n')
assert all(d['ok'] for d in designs)
a,b=[d['data'] for d in designs]
files=[Path(d['core_path']) for d in (a,b)]
check=dict(reference_files=[str(files[0])],candidate_files=[str(files[1])],top='dut',inputs=[dict(name='x',width=16)],outputs=[dict(name='y',width=24)],combinational_contract=True,timeout_s=120)
source_hashes=[fingerprint([path],'dut') for path in files]
wrong=folder/'wrong.sv';text=files[1].read_text();assert text.count('assign y=')==1
wrong.write_text(text.replace('assign y=',"assign y=24'h800000 ^ "))
sequential=folder/'sequential.sv';sequential.write_text('module dut(input clk,input [15:0] x,output reg [15:0] y); always @(posedge clk) y<=x; endmodule')
done={}

def call(op,payload):return dispatch(op,payload,cwd=root)

def rejected(op,payload):
 try:call(op,payload)
 except (InvalidInput,ValueError) as error:return dict(rejected=True,error_type=type(error).__name__,reason=str(error))
 raise AssertionError('Negative input unexpectedly accepted: '+op)

def execute(number):
 if number in done:return done[number]
 ident,paper,op,_=next(r for r in CATALOG if r[0]==number)
 destination=here/f'exp-tool-{ident}-{paper}'/'output'/('wide-'+key[:16]+'.json');destination.parent.mkdir(exist_ok=True)
 if destination.exists():
  result=json.loads(destination.read_text())
  assert result['source_sha256']==source_hashes
  done[number]=result;return result
 start=time.monotonic();positive=negative=None
 if number=='01':
  positive=call(op,dict(topics=['arithmetic']))
  assert any(rule['id']=='constant-multiply' for rule in positive['data']['rules'])
  negative=call(op,dict(topics=['not-a-known-topic']));assert not negative['data']['rules']
 elif number=='02':
  payload=dict(a_min=0,a_max=65535,b_min=255,b_max=255,operation='mul',declared_width=32)
  positive=call(op,payload);assert positive['data']['required_width']==24
  negative=call(op,dict(payload,declared_width=23));assert negative['data']['overflow_possible']
 elif number=='03':
  positive=call(op,check);assert positive['ok'] and positive['data']['vectors']==65536
  negative=call(op,dict(check,candidate_files=[str(wrong)]));assert not negative['ok']
 elif number=='10':
  positive=call(op,dict(check,constraint_ns=config['constraint_ns'],measure_timeout_s=config['measure_timeout_s']))
  negative=call(op,dict(check,candidate_files=[str(wrong)]))
  assert not negative['ok'] and negative['data']['stage']=='correctness' and len(negative['data']['children'])==1
 elif number=='04':
  evaluation=execute('10')
  positive=dict(ok=evaluation['positive']['ok'],parent=evaluation['positive']['run_id'],records=evaluation['positive']['data'].get('records',[]),children=evaluation['positive']['data']['children'])
  negative=rejected(op,dict(files=[str(sequential)],top='dut',name='sequential-negative'))
 else:
  measured=execute('04')['positive']
  if not measured['ok']:
   positive=dict(ok=False,note='Dependency measurement failed; no fabricated analytics inputs.')
  else:
   records=measured['records'];pair=dict(baseline=records[0],candidate=records[1])
   assert [r['source_sha256'] for r in records]==source_hashes
   if number=='05':
    positive=call(op,pair);bad=copy.deepcopy(pair);bad['candidate']['part']='deliberately-mismatched-part'
    negative=call(op,bad);assert not negative['ok']
   elif number=='06':
    positive=call(op,pair);bad=copy.deepcopy(pair);bad['candidate']['units']['delay_ns']='ps'
    negative=rejected(op,bad)
   elif number=='07':
    positive=call(op,dict(records=records))
    negative=rejected(op,dict(records=[records[0],records[0]]))
   elif number=='08':
    positive=call(op,dict(pair,correctness_passed=True,weights=dict(luts=.5,delay_ns=.5)))
    negative=call(op,dict(pair,correctness_passed=False));assert not negative['data']['eligible'] and negative['data']['reward'] is None
   elif number=='09':
    reward=execute('08')['positive']['data']['reward'];cost=max(.01,execute('10')['elapsed_s'])
    candidates=[dict(name='repeat-csd-evaluation',mean_reward=reward,visits=1,estimated_cost_s=cost)]
    positive=call(op,dict(candidates=candidates,budget_s=cost*1.1));assert positive['data']['selected']=='repeat-csd-evaluation'
    negative=call(op,dict(candidates=candidates,budget_s=cost*.1));assert negative['data']['selected'] is None
 assert source_hashes==[fingerprint([path],'dut') for path in files]
 result=dict(tool_id=ident,operation=op,input_sha256=key,source_sha256=source_hashes,config=config,environment=environment,positive=positive,negative=negative,elapsed_s=time.monotonic()-start,
   scope='Tool integration on existing CSD16. Combinational timing is not used for registered physical acceptance. Negative records are explicitly mutated controls.')
 destination.write_text(json.dumps(result,indent=2)+'\n');done[number]=result
 print(number,op,positive['ok'],destination.name,flush=True)
 return result

for row in CATALOG:
 if args.tool in ('all',row[0]):execute(row[0])
(folder/('summary.json' if args.tool=='all' else f'summary-{args.tool}.json')).write_text(json.dumps(list(done.values()),indent=2)+'\n')
sys.exit(0 if all(result['positive']['ok'] for result in done.values()) else 1)
