"""Compare native FP32 head kernels for full prefill and the last row."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root))
from src.runtime.compiled import NativeOrtOperators
deploy=root/'exp/1007_sw_stack/deploy'
r=deploy/'runs/compiled-diagnose-02'
index=json.loads((r/'reference-full-index.json').read_text())
x=np.load(r/index['/model/layers.30/final_norm_layernorm/output_0'])
w=np.memmap(r/'0301-MatMul-input1.bin',np.float32,mode='r',shape=(576,49152))
ops=NativeOrtOperators(deploy/'packages/onnx-ops',deploy/'packages/onnx-runtime/libonnx_ort_bridge.so')
ops.sessions['MatMul']=ops.lib.onnx_ort_create(str(root/'model-assets/official-compiled/float-ops/MatMul.onnx').encode())
with np.load(deploy/'data/official-oracle/reference.npz') as f: expected=f['logits'][0].copy()
for name,a in [('full-prefill',x),('last-row',x[:,-1:])]:
    y=ops('MatMul',[a,w])[0,-1]
    print(name,'token',int(y.argmax()),'oracle_error',float(np.abs(y-expected).max()),flush=True)
    np.save(r/f'head-{name}.npy',y)
ops.close()
