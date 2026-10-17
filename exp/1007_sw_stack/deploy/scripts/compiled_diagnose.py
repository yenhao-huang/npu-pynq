"""Capture the first compiled forward and the original graph's first layer."""
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np

root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root))
from src.inference.engine import LLMEngine
from src.inference.engine import ONNXReferenceEngine
from src.runtime.compiled import NativeOrtOperators
from acceptance import PROMPT
deploy=root/'exp/1007_sw_stack/deploy'
run=deploy/'runs/compiled-diagnose-02'
run.mkdir(parents=True,exist_ok=True)
package=root/'model-assets/official-compiled-debug-v3'
if not package.exists():
    shutil.copytree(root/'model-assets/official-compiled',package,copy_function=os.link)
    manifest=json.loads((package/'manifest.json').read_text())
    (package/'manifest.json').unlink()
    manifest['libraries']['pynq']='pynq/official_debug.so'
    (package/'manifest.json').write_text(json.dumps(manifest))
    shutil.copy2('/home/xilinx/jupyter_notebooks/smollm-compiled-stage-20261009/official_debug.so',package/'pynq/official_debug.so')
os.environ['NPU_OFFICIAL_DEBUG_DIR']=str(run)
from pynq import Overlay
overlay=Overlay(str(deploy/'hardware/overlay/npu_matrix.bit'),download=False)
engine=LLMEngine(package,overlay=overlay)
ids=engine.tokenizer.encode(engine.tokenizer.chat([{'role':'user','content':PROMPT}]))
engine.prefill(ids)
print('Compiled first token',engine.logits.argmax(),flush=True)
for b in (engine.model.io,engine.model.stage,engine.model.words):
    if b is not None: b.buffer.freebuffer()
del engine
ops=NativeOrtOperators(deploy/'packages/onnx-ops',deploy/'packages/onnx-runtime/libonnx_ort_bridge.so')
reference=ONNXReferenceEngine(deploy,float_ops=ops)
nodes=reference.graph.manifest['nodes']
end=next(i for i,n in enumerate(nodes) if n['outputs'][0]=='/model/layers.1/input_layernorm/output_0')
reference.graph.manifest['nodes']=nodes[:end+1]
names=set(n for node in nodes[:end+1] for n in node['outputs']
          if n and ('/model/layers.' in n or n=='/model/embed_tokens/Gather/output_0'))
reference.graph.manifest['outputs']=list(names)
captured={name:None for name in names}
reference.graph.run({'input_ids':np.asarray([ids],np.int64),
 'attention_mask':np.ones((1,len(ids)),np.int64),
 'position_ids':np.arange(len(ids),dtype=np.int64)[None],**reference.past},capture=captured)
index={}
for i,(name,value) in enumerate(captured.items()):
    if value is None: continue
    np.save(run/f'reference-{i:03d}.npy',value)
    index[name]=f'reference-{i:03d}.npy'
(run/'reference-index.json').write_text(json.dumps(index,indent=2))
print('Saved first-layer reference',flush=True)
