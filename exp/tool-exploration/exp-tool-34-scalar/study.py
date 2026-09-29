"""Substantive FIFO protocol experiment; physical acceptance is separate."""
import json
import os
from pathlib import Path
from ic_core import dispatch
here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
config=json.loads((here/'config.json').read_text())
out=here/'output';out.mkdir(exist_ok=True)
rows=[]
for params in config['configurations']:
    for architecture in config['architectures']:
        generated=dispatch('synth_fifo',dict(params,architecture=architecture))
        checked=dispatch('sequential_scoreboard',dict(params,files=generated['data']['files'],random_cycles=8192,seed=928))
        assert checked['data']['source_sha256']==generated['data']['source_sha256']
        row=dict(configuration=params,architecture=architecture,generator=generated,scoreboard=checked)
        rows.append(row)
        (out/(checked['run_id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        print(params,architecture,checked['ok'],checked['run_id'],flush=True)
        assert checked['ok']
