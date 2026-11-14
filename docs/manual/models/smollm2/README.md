# SmolLM2-135M-Instruct

The language model the SmolLM example runs on the PYNQ-Z1
([examples/smollm-safetensors](../../../../examples/smollm-safetensors/README.md)).
It is SmolLM2, Hugging Face's second SmolLM generation, in its smallest,
instruction-tuned size. The example folder and the CLI call it `smollm` for
short.

## Where it comes from

| | |
| --- | --- |
| Repository | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| Revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` (pinned) |
| License | Apache-2.0 |
| Pin record | [examples/smollm-safetensors/model-source.json](../../../../examples/smollm-safetensors/model-source.json): URL, byte length and SHA-256 per file |

Download the four files of that revision into the ignored checkpoint folder:

```bash
curl --fail --location --retry 3 --create-dirs \
  --output "examples/smollm-safetensors/model/checkpoint/#1" \
  "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/{config.json,tokenizer.json,tokenizer_config.json,model.safetensors}"
```

| File | Bytes | What it is |
| --- | --- | --- |
| `config.json` | 861 | architecture (below) |
| `tokenizer.json` | 2,104,556 | BPE tokenizer, copied into the package |
| `tokenizer_config.json` | 3,764 | chat template and special tokens |
| `model.safetensors` | 269,060,552 | weights, bfloat16 |

## Architecture

A Llama-style decoder (`LlamaForCausalLM`), from `config.json`:

| | |
| --- | --- |
| Parameters | 134,515,008 (272 tensors) |
| Layers | 30 |
| Hidden size | 576 |
| Attention | 9 query heads, 3 key/value heads (grouped-query), head size 64 |
| MLP | SiLU-gated, intermediate size 1536 |
| Normalization | RMSNorm, eps 1e-5 |
| Position | RoPE, theta 100,000; trained context 8192 |
| Vocabulary | 49,152; input embedding and output head share weights |
| Special tokens | bos 1, eos 2; the runtime also stops at `<\|im_end\|>` and `<\|endoftext\|>` |

Per layer the contractions are q and o (576 x 576), k and v (576 x 192),
gate and up (576 x 1536) and down (1536 x 576).

## How this repository uses it

[examples/smollm-safetensors/export.yml](../../../../examples/smollm-safetensors/export.yml)
drives the export:

| Setting | Value | Effect |
| --- | --- | --- |
| `load` | `huggingface`, the checkpoint folder | `AutoModelForCausalLM.from_pretrained` in float32 |
| `quantization` | `w8a8` | every linear layer and the embedding become INT8 weights with per-channel scales; activations are quantized per token at run time |
| `graph.max_seq` | 256 | KV cache slots: prompt plus answer cannot exceed 256 tokens (the model itself was trained to 8192) |
| `graph.chunk` | 16 | tokens per prefill call |
| `devices` | `cpu`, `npu` | the INT8 matrix multiplications run on the NPU, everything else on the Cortex-A9 |

The export produces two programs with static shapes: `decode` (one token)
and `prefill` (16 tokens). With 256 slots, one KV cache set is
30 layers x 2 x (3 x 256 x 64) float32 = 11.8 MB, and the runtime keeps two
sets that it swaps on every call.

On the PYNQ-Z1 (2026-11-01) prefill took about 48 s and decode ran at about
0.5 tokens/s, with output identical to the host simulator.
