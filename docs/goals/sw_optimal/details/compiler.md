# MLIR + LLVM compiler (#121)

```text
model frontend (src/models/<model>) -> linalg-on-tensors MLIR + npu.weight     (src/compiler/ops)
  npu-opt --npu-partition    INT8 contractions on weights -> npu.matmul         = NPU task
                             everything else stays linalg/arith/math/scf        = CPU task
  pack weights               NPU-read weights as B tiles (INT4: two per byte)
  npu-opt --npu-lower        npu.matmul -> ISA program(s) + call @npu_rt_gemm   = NPU instruction encoder
  mlir-opt                   fuse elementwise, one-shot-bufferize, dealloc, loops, LLVM dialect
  mlir-translate, llc        LLVM IR -> armv7a hard-float NEON (Cortex-A9) | arm64 host
  link                       + src/runtime/c/npu_rt.c -> pynq/model.so | host/model.dylib
```

## The npu dialect and passes (`src/compiler/mlir`)

- `npu.weight %arena "name"`: a named tensor in the weight arena. The exporter
  decides the bytes after partitioning, so a weight only the NPU reads is
  stored tiled and any other dense.
- `npu.matmul %lhs, %rhs {tk}` -> `tensor<Kt x M x N x i32>`: the K-chunked
  partial products. Keeping the planes apart is what makes group-wise INT4
  exact on an INT8 array: the CPU scales each plane by its group's scale.
- `--npu-partition` matches `linalg.matmul` (i8 x i8 -> i32, rhs a weight)
  and the group-wise contraction tagged `npu.group_size`, and writes a JSON
  report (NPU tasks; remaining CPU linalg ops).
- `--npu-lower` plans tiles (C++ port of `src/isa/layout.py`), splits a task
  into column *segments* of at most 4 MiB of weights (each its own program
  over a contiguous block), records ADDR_B relocations, and replaces the
  task by `npu_rt_gemm(id, A, partials)`.

## Execution model

The board runtime (`src/runtime/compiled.py` + `npu_rt.c`) puts programs and
tiles in CMA, relocates every ADDR_B word to where its weight block lives,
and per call packs A, runs each segment program on the NPU (or the C ISA
simulator on the host), and unpacks the INT32 planes. Weights are either
resident in CMA or memory-mapped and streamed segment by segment into a
small stage window (INT4 blocks are widened on the way). The same package
runs on both platforms; only the shared library differs.

## What lands where (W8A8 packages)

| Model | NPU tasks / segments | CPU linalg ops | Weights | Export time |
| --- | --- | --- | --- | --- |
| SmolLM2-135M | 242 / 254 | 2484 | 136 MB | 17 s (AWQ) |
| Qwen3-0.6B | 226 / 356 | 2768 | 598 MB | 8 s |
| ResNet-18 | 21 / 21 | 196 | 12 MB | 1 s |

(`exp/1007_sw_stack/host/host_acceptance.json`; two entry points each for the
LLMs, decode and prefill.)

## Tests

`src/test/tests/test_compiler.py`: W8/W4 MLPs equal NumPy on the simulated
NPU; C++ encoder words equal the Python encoder for every segment; the board
library is ELF32 ARM hard-float; packed INT4 streamed and INT8 resident vs
streamed agree. They skip when LLVM 22 with MLIR is absent.
