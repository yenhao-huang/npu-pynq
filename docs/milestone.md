# Milestone: software stack features

Status of the inference framework (`src/runtime/`, `src/vendors/npu_pynq/driver/`)
and the compiler (`src/export/`, `src/vendors/npu_pynq/custom_dialect/`), checked
against features common in mainstream inference frameworks and ML compilers
(vLLM, TensorRT-LLM, llama.cpp, ExecuTorch, TVM, XLA, MLIR).

- `[x]` done
- `[ ]` planned. Items marked *(proposed)* are suggestions not yet in
  `docs/plans/2026-1029-002-export-pytorch-exportedprogram.md`.

Not done and not planned: PagedAttention, continuous batching, server/API,
QAT/calibration, auto-tuning, dynamic shapes.

## Inference framework

### Quantization
- [x] Weight quantization (W8A8)
- [ ] Weight quantization (W4A8)
- [ ] KV cache quantization *(proposed)*

### Memory and caching
- [x] KV cache
- [x] Weight streaming from mmap
- [ ] Resident weights
- [ ] Prefix caching *(proposed)*

### Generation
- [x] Chunked prefill
- [x] Sampling strategies
- [x] Streaming output
- [ ] Speculative decoding *(proposed)*

### Hardware execution
- [x] Heterogeneous offload (delegates)
- [x] Simulator backend
- [x] Hardware compatibility check
- [x] Performance statistics

### Model support
- [x] SmolLM2, ResNet-18
- [ ] Qwen family

## Compiler

### Front end
- [x] Graph capture
- [x] Multi-level IR

### Quantization
- [x] PTQ (W8A8)
- [ ] PTQ (W4A8)

### Graph optimization
- [x] Constant folding (partial)
- [x] CSE / DCE
- [x] Operator fusion (CPU)
- [ ] Operator fusion (q/k/v and gate/up projections)

### Hardware mapping
- [x] Partition / delegate
- [x] Layout transformation
- [x] Tiling
- [ ] Cascading / double buffering *(proposed)*

### Memory
- [x] Static memory planning (partial)
- [ ] Static memory planning (complete) *(proposed)*

### Codegen
- [x] Multi-target cross-compilation
- [ ] Multi-target cross-compilation (macOS)
- [x] Direct object emission (partial: runs `llc`)
- [ ] Direct object emission (LLVM `TargetMachine`)
- [ ] Vectorization
- [ ] Custom kernel registration

### Testing
- [ ] Per-pass unit tests
