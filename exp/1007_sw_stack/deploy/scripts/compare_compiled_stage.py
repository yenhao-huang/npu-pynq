"""Locate the first divergence among captured first-layer native inputs."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1])
index=json.loads((root/'reference-index.json').read_text())
pairs=[
 ('0000-SimplifiedLayerNormalization-input0.bin','/model/embed_tokens/Gather/output_0'),
 ('0001-DynamicQuantizeLinear-input0.bin','/model/layers.0/input_layernorm/output_0'),
 ('0002-RotaryEmbedding-input0.bin','/model/layers.0/attn/q_proj/MatMul/output_0'),
 ('0003-RotaryEmbedding-input0.bin','/model/layers.0/attn/k_proj/MatMul/output_0'),
 ('0004-GroupQueryAttention-input0.bin','/model/layers.0/attn/q_rotary/RotaryEmbedding/output_0'),
 ('0004-GroupQueryAttention-input1.bin','/model/layers.0/attn/k_rotary/RotaryEmbedding/output_0'),
 ('0004-GroupQueryAttention-input2.bin','/model/layers.0/attn/v_proj/MatMul/output_0'),
 ('0005-DynamicQuantizeLinear-input0.bin','/model/layers.0/attn/GroupQueryAttention/output_0'),
 ('0006-SkipSimplifiedLayerNormalization-input0.bin','/model/embed_tokens/Gather/output_0'),
 ('0006-SkipSimplifiedLayerNormalization-input1.bin','/model/layers.0/attn/o_proj/MatMul/output_0'),
 ('0007-DynamicQuantizeLinear-input0.bin','/model/layers.0/post_attention_layernorm/output_0'),
 ('0008-Sigmoid-input0.bin','/model/layers.0/mlp/gate_proj/MatMul/output_0'),
 ('0009-DynamicQuantizeLinear-input0.bin','/model/layers.0/mlp/Mul/output_0'),
 ('0010-SkipSimplifiedLayerNormalization-input1.bin','/model/layers.0/mlp/down_proj/MatMul/output_0'),
 ('0011-DynamicQuantizeLinear-input0.bin','/model/layers.1/input_layernorm/output_0')]
for actual,name in pairs:
    a=np.fromfile(root/actual,np.float32)
    b=np.load(root/index[name]).reshape(-1)
    print(actual,'max_error',float(np.abs(a-b).max()),'equal',bool(np.array_equal(a,b)),
          'actual',a[:4].tolist(),'expected',b[:4].tolist(),flush=True)
