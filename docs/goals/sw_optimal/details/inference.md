# Inference framework and architectures (#122)

Everything the models compute is compiled; Python only schedules.

| Piece | Where | Evidence |
| --- | --- | --- |
| Weight loading (safetensors, bf16) | `src/inference/weights.py` | unit test |
| Tokenizer (byte-level BPE, special tokens, NFC, Digits) | `src/inference/tokenizer.py` | identical ids to HF `tokenizers` on 9 tricky strings, SmolLM2 and Qwen3.5, with and without `regex` |
| Prefill / decode | `src/inference/graph.py`, `engine.py` | separate entry points; prefill runs 16-token causal chunks; prefill and decode logits identical (max diff 0.0) |
| KV cache | `src/compiler/ops/attention.py` | FP32 or INT8 (+ per position/head scale) |
| Sampling | `src/inference/sampling.py` | greedy, temperature, top-k, top-p, min-p, repetition penalty |

## Architectures (all operators shared, each model its own package)

| Feature | Operator | Validated on |
| --- | --- | --- |
| Tied embedding | `embedding.tied_lookup` (reads rows from lm_head's NPU tile layout: one copy) | SmolLM2, Qwen3 |
| RMSNorm (+ zero-centred) | `core.rmsnorm` | all LLMs |
| RoPE (full, partial) | `embedding.rope` | Qwen3.5 rotates 64 of 256 dims |
| GQA / MQA | `attention` (kv head = q head // group) | 9/3, 16/8, 8/2 heads; MQA in unit test |
| Attention, two implementations | `attention(mode="naive" \| "online")` | 8 variants equal NumPy |
| SwiGLU MLP | `core.silu_mul` + fused gate/up task | all LLMs |
| MoE | `moe.moe` (FP32 router, rank top-k, scf.if expert dispatch) | Qwen3-MoE |
| Linear attention | `linear_attention.gated_delta` (Gated DeltaNet) | Qwen3.5 |

FP32 builds against Transformers (`exp/1007_sw_stack/host/arch_check*.py`):

| Model | Layers checked | Max relative logit error | Top-1 |
| --- | --- | --- | --- |
| SmolLM2-135M | 30 (full) | cosine 1.0, 24/24 greedy tokens x 3 prompts | 1.0 |
| Qwen3-0.6B | 2 | 2.2e-6 | 1.0 |
| Qwen3-MoE (seeded 4-layer, 8 experts, top-2) | 4 | 1.2e-6 | 1.0 |
| Qwen3.5-0.8B (3 Gated DeltaNet + 1 gated attention) | 4 | 2.7e-6 | 1.0 |

MoE dispatch is sparse: one decode step issues 23 NPU calls (routed experts
only) instead of 59.

## Host runs (Apple M-series, NPU simulated in C; `host_acceptance.json`)

| Model | Prompt "What is the capital of France?" | Decode tok/s | Prefill tok/s |
| --- | --- | --- | --- |
| SmolLM2-135M W8A8 + AWQ | "The capital of France is Paris. Paris is a city located in the northern part of the country..." | 22.7 | 30.0 |
| Qwen3-0.6B W8A8 | "&lt;think&gt; Okay, the user is asking for the capital of France. I know that France's capital is Paris..." | 5.0 | 6.7 |
| Qwen3.5-0.8B W8A8 | "Paris is the capital of France. ..." (completion prompt) | 4.3 | 2.3 |

ResNet-18 W8A8: 5/5 gallery images correct, top-1 equal to TorchVision FP32,
logit cosine >= 0.9986, 0.6 s per image (`resnet18_vs_torchvision.json`).
