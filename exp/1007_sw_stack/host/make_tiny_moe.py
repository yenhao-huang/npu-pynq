"""Write a small randomly initialised Qwen3-MoE checkpoint for architecture tests.

No public Qwen3-MoE fits a laptop test loop or the PYNQ-Z1, so the MoE path
is validated on a seeded random model against Transformers' own forward.

    python exp/1007_sw_stack/host/make_tiny_moe.py build/tiny-qwen3-moe
"""

import sys

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

out = sys.argv[1] if len(sys.argv) > 1 else "build/tiny-qwen3-moe"
torch.manual_seed(0)
config = Qwen3MoeConfig(vocab_size=1024, hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2, head_dim=32,
                        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, mlp_only_layers=[3],
                        max_position_embeddings=512, tie_word_embeddings=False)
model = Qwen3MoeForCausalLM(config)
with torch.no_grad():
    for p in model.parameters():
        if p.dim() > 1:
            p.normal_(0.0, 0.06)
model.save_pretrained(out)
print(out)
