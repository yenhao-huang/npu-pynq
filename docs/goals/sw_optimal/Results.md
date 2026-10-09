# Results: software-stack optimization (#119)

Goal: [goal.md](goal.md). How to rerun everything: [Reproduce.md](Reproduce.md).

## What was built

```text
compile:  model (src/models/*) -> MLIR (linalg + npu dialect) -> npu-partition
            |- CPU task -> bufferize -> LLVM IR -> llc (Cortex-A9, NEON)  \
            `- NPU task -> NPU instruction encoder -> ISA program          > one package
run:      Cortex-A9 code runs the CPU work and calls npu_rt, which hands
          programs + tiles to the FPGA: IF fetches, ID decodes, EX runs the
          16 x 16 array and writes results back to DDR for the ARM
```

| Part | Branch / PR | Key result | Details |
| --- | --- | --- | --- |
| Custom NPU ISA, IF + ID stages | `npu/issue120-a`, #123 | 10 instructions; front end fetches/decodes from DDR via its own AXI4 master; 14/14 testbenches incl. 7 encoder programs checked word-for-word | [hardware-isa.md](details/hardware-isa.md) |
| MLIR + LLVM compiler | `npu/issue121-a`, #124 | `npu` dialect + partition/lower passes; ARM hard-float and host code from the same IR | [compiler.md](details/compiler.md) |
| Inference framework, models, quantization | `npu/issue122-a`, #125 | tokenizer, prefill/decode, KV cache, sampling; GQA, two attention kernels, MoE, linear attention; W8A8, W4A8, AWQ, INT8 KV | [inference.md](details/inference.md), [quantization.md](details/quantization.md) |

## Acceptance

| Model | Export | Host (simulated NPU) | PYNQ-Z1 NPU |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct | PASS: W8A8 + AWQ, 242 NPU tasks, 136 MB | PASS: "The capital of France is Paris...", 22.7 tok/s; FP32 build equals Transformers (24/24 greedy tokens) | see [board.md](details/board.md) |
| Qwen3-0.6B | PASS: W8A8 598 MB; board build W4A8 g64 + AWQ, 398 MB streamed | PASS: answers "Paris" (thinking mode), 5.0 tok/s; FP32 equals Transformers (rel err 2e-6) | see [board.md](details/board.md) |
| ResNet-18 | PASS: W8A8, 21 NPU tasks, 12 MB | PASS: 5/5 gallery images, top-1 = TorchVision FP32 | see [board.md](details/board.md) |

Further architectures on the same operators: Qwen3-MoE (sparse expert
dispatch, rel err 1.2e-6) and Qwen3.5-0.8B (Gated DeltaNet linear attention,
rel err 2.7e-6).

## Quantization (perplexity vs FP32, 256 tokens)

| Model | FP32 | W8A8 | W8A8 + AWQ | W4A8 | W4A8 + AWQ |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 8.78 | 9.13 | **8.79** | 12.11 | **10.35** |
| Qwen3-0.6B | 9.65 | 9.78 | 9.80 | 11.84 | **11.04** |

The compiled W8A8 packages are bit-exact with fake-quantized Transformers
wherever the float paths coincide; see [quantization.md](details/quantization.md).

## Open items

- Board columns: [board.md](details/board.md) records the physical runs.
- Vivado timing of the ISA overlay at 100 MHz is reported there too.
