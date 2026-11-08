"""Report the first nonidentical native input across all compiled layers."""
import json
from pathlib import Path
import sys
import numpy as np
r=Path(sys.argv[1]); index=json.loads((r/'reference-full-index.json').read_text())
for i in range(30):
    prefix=f'/model/layers.{i}/'
    pairs=[(1,'DynamicQuantizeLinear',0,prefix+'input_layernorm/output_0'),
     (2,'RotaryEmbedding',0,prefix+'attn/q_proj/MatMul/output_0'),
     (3,'RotaryEmbedding',0,prefix+'attn/k_proj/MatMul/output_0'),
     (4,'GroupQueryAttention',0,prefix+'attn/q_rotary/RotaryEmbedding/output_0'),
     (4,'GroupQueryAttention',1,prefix+'attn/k_rotary/RotaryEmbedding/output_0'),
     (4,'GroupQueryAttention',2,prefix+'attn/v_proj/MatMul/output_0'),
     (5,'DynamicQuantizeLinear',0,prefix+'attn/GroupQueryAttention/output_0'),
     (6,'SkipSimplifiedLayerNormalization',0,'/model/embed_tokens/Gather/output_0' if i==0 else prefix+'input_layernorm/output_3'),
     (6,'SkipSimplifiedLayerNormalization',1,prefix+'attn/o_proj/MatMul/output_0'),
     (7,'DynamicQuantizeLinear',0,prefix+'post_attention_layernorm/output_0'),
     (8,'Sigmoid',0,prefix+'mlp/gate_proj/MatMul/output_0'),
     (9,'DynamicQuantizeLinear',0,prefix+'mlp/Mul/output_0'),
     (10,'SkipSimplifiedLayerNormalization',0,prefix+'post_attention_layernorm/output_3'),
     (10,'SkipSimplifiedLayerNormalization',1,prefix+'mlp/down_proj/MatMul/output_0')]
    for step,op,slot,name in pairs:
        a=np.fromfile(r/f'{10*i+step:04d}-{op}-input{slot}.bin',np.float32)
        b=np.load(r/index[name]).reshape(-1)
        error=float(np.abs(a-b).max())
        if error:
            print('FIRST DIFFERENCE',i,name,'max_error',error,'actual',a[:6].tolist(),
                  'expected',b[:6].tolist(),flush=True)
            raise SystemExit(1)
    print('Layer',i,'exact',flush=True)
a=np.fromfile(r/'0301-MatMul-input0.bin',np.float32)
b=np.load(r/index['/model/layers.30/final_norm_layernorm/output_0'])[:,-1].reshape(-1)
print('Final head input error',float(np.abs(a-b).max()))
