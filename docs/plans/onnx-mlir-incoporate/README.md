# ONNX-MLIR Compilation for SmolLM2

The pinned SmolLM2 INT8 model now exports to host and Cortex-A9 packages through
ONNX-MLIR import and the existing NPU compiler. Physical PYNQ-Z1 acceptance
passed prefill 36/40/42 and three decode steps each. The package has no ORT
operator sessions, ORT shared library, or manually compiled bridge object.

See [the plan](plan.md), [acceptance evidence](acceptance.md), and the
[WSL](../../../examples/smollm_onnx/README.md) /
[macOS](../../../examples/smollm_onnx/README.macos.md) deployment guides.

## Compiler boundary

```text
pinned ONNX
  -> ONNX-MLIR 0.5.1: generic ONNX dialect IR (LLVM 24)
  -> export-side MLIR 22 parser: operations, attributes, types and SSA edges
  -> specialization and verified CPU operator decomposition
  -> unchanged NPU partition: INT8 contractions with npu.weight
       -> unchanged NPU lowering, programs and weight tiling
       -> LLVM 22 CPU lowering and ARM32 hard-float code generation
  -> package
  -> compiled ARM control flow and existing NPU driver
```

The pinned importer image lacks ARM code generation. It only needs to import:
`--EmitONNXIR --mlir-print-op-generic` produces generic syntax that the MLIR 22
adapter parses successfully. CPU semantics lower through linalg/arith/math/SCF
on the export side. ARM code generation uses the existing ARM-enabled LLVM 22.
No vendor, ISA, driver, or RTL upgrade is needed.

ONNX custom operations need explicit conversions; importing them does not
establish executable semantics. Normalization, residuals, dynamic quantization,
Sigmoid, rotary embedding and GQA have numerical tests. Unsupported custom
attributes fail explicitly. The pinned source SHA-256 fixes the supported model
topology; this is not a general converter for arbitrary ONNX models.

## Requirements and pins

Use Python with NumPy/ONNX, CMake 3.24+, Ninja, LLVM/MLIR 22 development
libraries, clang and lld. Set `NPU_LLVM_BIN` when discovery needs an override.
The C++ parser adapter and unchanged `npu-opt` build on first use.

`src/export/compilers/onnx_mlir/toolchain.json` pins importer revision
`20f2b509a1edddcc38000b04bca5ab38c1f183f9`, version 0.5.1, LLVM 24 revision
`43574226b712f94de17f07392a64282ac8ddab23`, and this immutable image:

```bash
docker pull ghcr.io/onnxmlir/onnx-mlir@sha256:5df1c1d3f1d8d5207c9b408e9d6e0ec2ae9926c6948b4b5c745358a2b47ad2e8
```

Normal imports do not pull or access the network. Explicit native compiler
overrides retain unverified provenance.

## Production export

```bash
export NPU_ONNX_MODEL="$PWD/dist/smollm2-onnx"
python -m src.export.cli prepare-onnx fetch
python -m src.export.cli export smollm \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --out dist/smollm-package --prefill-lengths 36,40,42
```

The exporter imports automatically and caches imports under ignored `build/`.
It verifies source and IR hashes before parsing and partitioning. Keep adjacent
config/tokenizer files with the pinned source.

To retain a separate, inspectable import boundary:

```bash
python -m src.export.compilers.onnx_mlir.pipeline \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --out build/smollm-onnx-import
python -m src.export.cli export smollm \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --onnx-import-dir build/smollm-onnx-import \
  --out dist/smollm-package --prefill-lengths 36,40,42
```

Import output directories must be new or empty. Inventory, compiler logs,
generic IR and the import report remain available after failures. The standalone
import report deliberately says `compile_ready: false`: it reports its import
stage, not completion of target compilation or board validation.

`prepare-onnx prepare` fetches, exports a reference JSON/tensor graph, and verifies
its initializers. It does not create MLIR or require ORT session assets.
`prepare-onnx operators` separately prepares optional host ORT reference sessions.

## Numerical contract

Original INT8 weights, unsigned activation zero-point corrections, FP32 KV
storage and entry-point ABI are retained. All 480 NPU tasks, program bytes,
weight offsets, layouts and the weight blob match the legacy frontend.

Quantized graphs are sensitive to rounding at half-integer thresholds.
Conversions preserve normalization division order, exact constant formatting,
explicit fused arithmetic, fixed softmax accumulation and decode grouping.
The shared literal formatter preserves the Python double until MLIR rounds it
to its destination type. It previously truncated a coefficient across an FP32
midpoint. Tests cover that regression and full-width prefill/decode attention.

The CPU math is generated MLIR. Calls to its private helper, libm and the
existing NPU driver are compiled calls; they create no ORT sessions.

## Independent host and board acceptance

Host-only reference generation needs `onnxruntime==1.20.1`. It executes the
unchanged source ONNX, independently of the candidate compiler:

```bash
python -m src.test.onnx_mlir_acceptance fixtures \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --package dist/smollm-package --out build/smollm-reference
python -m src.test.onnx_mlir_acceptance check \
  --package dist/smollm-package --fixtures build/smollm-reference \
  --evidence build/smollm-host.json
```

Copy the reference directory and `src/test/onnx_mlir_acceptance.py` with the
runtime when performing numerical board acceptance. On the board:

```bash
sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
  -m src.test.onnx_mlir_acceptance check --backend pynq \
  --package smollm-package --fixtures smollm-reference \
  --overlay npu_matrix.bit --evidence smollm-board.json
```

This checks hashes, logits, KV state, greedy tokens, positions, untouched cache
regions, memory and physical NPU counters. The board does not need ORT.
Prefill lengths are exact full-chat input lengths because quantization spans
the entire prompt.

## Regression commands

```bash
python -m unittest src.test.tests.test_onnx_import \
  src.test.tests.test_onnx_lowering src.test.tests.test_onnx_adapter \
  src.test.tests.test_compiler src.test.tests.test_ops \
  src.test.tests.test_vendor_boundary src.test.tests.test_jit_boundary \
  src.test.tests.test_isa -v
NPU_TEST_ONNX_MLIR=1 python -m unittest src.test.tests.test_onnx_import -v
```

LLVM-dependent tests must run with LLVM/MLIR 22 installed. Generated IR, compiler
builds, packages, reference tensors and executed notebooks remain untracked.
