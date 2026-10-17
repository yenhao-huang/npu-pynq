"""Check the captured native head weights against the official tied embedding."""
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root))
from src.inference.graph import ONNXGraph
r=root/'exp/1007_sw_stack/deploy/runs/compiled-diagnose-02'
g=ONNXGraph(root/'exp/1007_sw_stack/deploy/packages/onnx-graph')
w=g.weights
a=np.memmap(r/'0301-MatMul-input1.bin',np.float32,mode='r',shape=(576,49152))
q=w['model.embed_tokens.weight_quantized']; scale=w['model.embed_tokens.weight_scale']; zero=w['model.embed_tokens.weight_zero_point']
maximum=0
for start in range(0,576,16):
    expected=(q[:,start:start+16].astype(np.float32)-zero)*scale
    maximum=max(maximum,float(np.abs(a[start:start+16]-expected.T).max()))
print('Head weight error',maximum,'actual',a[0,:6].tolist(),flush=True)
