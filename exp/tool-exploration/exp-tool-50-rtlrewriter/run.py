"""Audit every current declaration and committed study, including failures."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser=argparse.ArgumentParser()
parser.add_argument('--study',action='append',default=[])
args=parser.parse_args()
payload=json.loads((here/'config.json').read_text())
payload['additional_studies']=args.study
result=dispatch('acceptance_audit',payload)
out=here/'output';out.mkdir(exist_ok=True)
(out/('audit-'+result['run_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(result['run_id'],json.dumps({k:result['data'][k] for k in ['gates','declared_cases','cases_with_verified_physical_pairs','qualifying_family_configurations','all_case_geometric_benefit_lower_bound','unsupported_declarations']}),flush=True)
