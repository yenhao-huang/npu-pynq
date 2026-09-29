"""Reproduce cycle validation across substantial arithmetic architectures."""
import json
import os
from pathlib import Path
from ic_core import dispatch
here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
config=json.loads((here/'config.json').read_text())
results=[]
for width in config['widths']:
    for architecture in config['architectures']:
        for operation in ('synth_serial_multiplier','synth_divider'):
            generated=dispatch(operation,dict(width=width,architecture=architecture))
            d=generated['data']
            checks=[]
            for fixture in (False,True):
                checks.append(dispatch('latency_throughput',dict(files=d['timing_files'] if fixture else d['files'],top=d['timing_top'] if fixture else d['top'],width=width,operation=d['operation'],latency_cycles=d['latency_cycles'],expected_ii=d['initiation_interval'],timing_fixture=fixture,random_cycles=config['random_cycles'])))
            results.append(dict(generator=generated,checks=checks))
out=here/'output';out.mkdir(exist_ok=True)
(out/'cycles.json').write_text(json.dumps(results,indent=2)+'\n')
print('cycle checks',sum(c['ok'] for r in results for c in r['checks']),'/',2*len(results))
