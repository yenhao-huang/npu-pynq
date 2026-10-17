"""Capture selected original graph values across all layers for diagnosis."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root))
from src.inference.engine import ONNXReferenceEngine
from src.runtime.compiled import NativeOrtOperators
from acceptance import PROMPT
deploy=root/'exp/1007_sw_stack/deploy'
run=deploy/'runs'/sys.argv[1]
ops=NativeOrtOperators(deploy/'packages/onnx-ops',deploy/'packages/onnx-runtime/libonnx_ort_bridge.so')
npu=None
if '--physical-reference' in sys.argv:
    from pynq import Overlay
    from src.runtime.isa_runtime import NpuGemm
    npu=NpuGemm(Overlay(str(deploy/'hardware/overlay/npu_matrix.bit'),download=False))
engine=ONNXReferenceEngine(deploy,integer_gemm=npu,float_ops=ops)
suffixes=['input_layernorm/output_0','input_layernorm/output_3',
 'post_attention_layernorm/output_0','post_attention_layernorm/output_3',
 'attn/q_proj/MatMul/output_0','attn/k_proj/MatMul/output_0','attn/v_proj/MatMul/output_0',
 'attn/q_rotary/RotaryEmbedding/output_0','attn/k_rotary/RotaryEmbedding/output_0',
 'attn/GroupQueryAttention/output_0','attn/o_proj/MatMul/output_0',
 'mlp/gate_proj/MatMul/output_0','mlp/Mul/output_0','mlp/down_proj/MatMul/output_0',
 'final_norm_layernorm/output_0']
capture={name:None for node in engine.graph.manifest['nodes'] for name in node['outputs']
         if name=='/model/embed_tokens/Gather/output_0' or any(name.endswith(s) for s in suffixes)}
ids=engine.tokenizer.encode(engine.tokenizer.chat([{'role':'user','content':PROMPT}]))
result=engine.graph.run({'input_ids':np.asarray([ids],np.int64),
 'attention_mask':np.ones((1,len(ids)),np.int64),
 'position_ids':np.arange(len(ids),dtype=np.int64)[None],**engine.past},capture=capture)
index={}
for i,(name,value) in enumerate(capture.items()):
    np.save(run/f'full-reference-{i:03d}.npy',value)
    index[name]=f'full-reference-{i:03d}.npy'
(run/'reference-full-index.json').write_text(json.dumps(index,indent=2))
print('Original first token',int(result['logits'][0,-1].argmax()),flush=True)
if npu:
    print('Original graph physical integer stats',npu.stats,flush=True)
    npu.close()
