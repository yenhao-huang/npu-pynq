# Results: software-stack optimization (#119)

Goal: [goal.md](goal.md). How to rerun everything: [Reproduce.md](Reproduce.md).

Progress aligned on 2026-10-09 with PR #126 at
`778c90c5d8340c362972a627c18eb27580d2251c`. The local follow-up branch
`npu/issue119-b` adds the official INT8 path below; it has not been pushed.

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

### Official SmolLM2 INT8 follow-up

- **Compiled physical acceptance PASS:** run `compiled-board-20261009-04`
  executed all eight notebook cells with 35/35 independent audit gates.
  ONNX frontend -> MLIR -> ISA lowering -> LLVM ARM -> `CompiledModel` is active;
  float kernels remain explicit native ORT 1.20.1 calls.
- Every logit and token in the 24-step trace exactly matches the unchanged full
  ARM ONNX oracle, separate ARM compiled ISA simulator and reset/repeat.
  Original quantization is retained; QKV/gate-up fusion reduces 210 contractions
  to 120 per forward. The run recorded 842,400 physical jobs and 453,076,654 cycles.
- TTFT **12.8602 s**, TPOT **1.5432 s/token**, throughput **0.6480 tokens/s**.
  These are recorded generation timings; initial smoke prefill was 26.62 s.
- **Shared source physical check PASS:** a separate process generated from a
  board directory containing only `src/`, with no `exp/`, using independent
  package and overlay assets. Seven focused Linux regressions passed.
- Exact compiled prefill lengths are 36, 40 and 42 chat tokens. Other lengths
  require compilation; arbitrary chunks would change official DQL semantics.

Details: [official-int8.md](details/official-int8.md) and
[CompiledResult.md](../../../exp/1007_sw_stack/deploy/CompiledResult.md).

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

- Complete the cancelled automated campaign for the original W8+AWQ SmolLM2,
  Qwen3 and ResNet-18 packages; the manual compiled official ONNX milestone
  does not substitute for that campaign.
- Collect routed 100 MHz timing evidence for the ISA overlay and completed
  automated board workflow results; see [board.md](details/board.md).
