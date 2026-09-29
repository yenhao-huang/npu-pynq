"""Checkpointed physical reduction study; run with repository virtualenv."""
import hashlib
import json
import os
from pathlib import Path
from ic_core import dispatch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'output'
digest=hashlib.sha256()
for source in sorted((ROOT/'tools/ic/ic_core').rglob('*.py')):
    digest.update(source.read_bytes())
digest.update(Path(__file__).read_bytes())
OUT=OUT/digest.hexdigest()[:16]
OUT.mkdir(exist_ok=True,parents=True)
os.chdir(ROOT)

def call(name, op, params):
    path=OUT/(name+'.json')
    if path.exists():
        cached=json.loads(path.read_text())
        if cached.get('ok'):
            return cached
    result=dispatch(op,params)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(name, result.get('ok'), flush=True)
    if not result.get('ok'):
        raise RuntimeError(f'{name}: {result}')
    return result

for width,lanes in [(16,16),(32,16)]:
    label=f'w{width}-n{lanes}'
    designs={}
    for arch in ['serial','balanced','compressor']:
        designs[arch]=call(label+'-'+arch+'-generate','synth_adder_tree',dict(width=width,lanes=lanes,architecture=arch))['data']
    for arch in ['balanced','compressor']:
        call(label+'-'+arch+'-check','vector_equivalence',dict(reference_files=[designs['serial']['core_path']],candidate_files=[designs[arch]['core_path']],top='dut',input_width=width*lanes,output_width=width,random_vectors=8192,combinational_contract=True))
    for repeat in range(3):
        for arch,data in designs.items():
            call(f'{label}-{arch}-r{repeat}','clocked_ppa',dict(files=[data['core_path'],data['wrapper_path']],top='registered_dut',name=f'{label}-{arch}',period_ns=5,latency_cycles=1,initiation_interval=1))
