"""Attribute real generated structures without treating estimates as routed PPA."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch
here=Path(__file__).resolve().parent
os.chdir(here.parents[1])
parser=argparse.ArgumentParser();parser.add_argument('--container');args=parser.parse_args()
folders={id:next(here.glob('exp-tool-'+id+'-*'))/'output' for id in ['39','40','41','42']}
for folder in folders.values(): folder.mkdir(exist_ok=True)
def save(id,label,result):
    (folders[id]/(label+'-'+result['run_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
for width,depth in [(16,64),(32,128)]:
    for architecture in ['shift','circular']:
        generated=dispatch('synth_fifo',dict(width=width,depth=depth,architecture=architecture))['data']
        for mapping in ['generic','xc7']:
            label=f'fifo-w{width}-d{depth}-{architecture}-{mapping}'
            result=dispatch('netlist_profile',dict(files=generated['files'],top='fifo_dut',mapping=mapping,container=args.container))
            save('39',label,result);print(label,result['ok'],result['run_id'],flush=True)
            if result['ok']:
                p=dict(netlist_file=result['data']['netlist_file'],top='fifo_dut',expected_sha256=result['data']['netlist_sha256'])
                for id,op in [('41','fanout_analysis'),('42','memory_inference')]: save(id,label,dispatch(op,p))
for width in [16,32]:
    for architecture in ['independent','shared']:
        generated=dispatch('synth_mcm',dict(width=width,architecture=architecture))['data']
        label=f'mcm-w{width}-{architecture}'
        result=dispatch('netlist_profile',dict(files=[generated['core_path']],top='dut',mapping='generic',container=args.container))
        save('39',label,result);print(label,result['ok'],result['run_id'],flush=True)
        if result['ok']:
            p=dict(netlist_file=result['data']['netlist_file'],top='dut',expected_sha256=result['data']['netlist_sha256'])
            save('40',label,dispatch('critical_cone',p))
            save('41',label,dispatch('fanout_analysis',p))
