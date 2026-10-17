# Quantization

| Scheme | Weights | Activations | Where |
| --- | --- | --- | --- |
| W8A8 | INT8, per output channel | INT8, per token (dynamic) | `ops/linear.py` |
| W4A8 | INT4, per output channel per K group (32-128) | INT8 per token | same; groups map to K chunks, so the NPU's partial planes are scaled exactly; stored two per byte |
| AWQ | per-input-channel scales searched on calibration activations, folded into RMSNorm / v / up weights | - | `inference/quant/awq.py` |
| KV cache | INT8 per (layer, head, position) | - | `ops/attention.py` |

The compiled packages reproduce a fake-quantized Transformers model
(`inference/quant/fakequant.py`) bit for bit wherever the float path is
identical (position 0 through all 30 SmolLM2 layers; every position with one
layer). Later positions differ only through INT8 rounding flips triggered by
attention's summation order, exactly as Transformers' own SDPA and eager
kernels differ from each other (top-1 0.91).

## Quality against FP32 (256 evaluation tokens, `awq_eval.py`, fake-quantized)

| Model | Scheme | Top-1 vs FP32 | KL | PPL (FP32) |
| --- | --- | --- | --- | --- |
| SmolLM2-135M | W8A8 | 0.916 | 0.031 | 9.13 (8.78) |
| | W8A8 + AWQ | 0.946 | 0.0095 | **8.79** |
| | W4A8 g64 | 0.774 | 0.324 | 12.11 |
| | W4A8 g64 + AWQ | 0.795 | 0.183 | **10.35** |
| Qwen3-0.6B | W8A8 | 0.949 | 0.025 | 9.78 (9.65) |
| | W8A8 + AWQ | 0.953 | 0.015 | 9.80 |
| | W4A8 g128 | 0.767 | 0.373 | 11.84 |
| | W4A8 g128 + AWQ | 0.826 | 0.237 | **11.04** |

## Compiled packages (200 tokens, `quant_eval.py`, SmolLM2 W8A8)

| KV cache / attention | Top-1 vs FP32 | KL | PPL (FP32 10.02) |
| --- | --- | --- | --- |
| FP32 KV, online softmax | 0.905 | 0.036 | 10.62 |
| FP32 KV, naive softmax | 0.920 | 0.035 | 10.45 |
| INT8 KV, online softmax | 0.910 | 0.035 | 10.83 |
| INT8 KV + AWQ | 0.940 | 0.010 | 10.26 |
