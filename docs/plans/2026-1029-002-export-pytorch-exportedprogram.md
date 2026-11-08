# Export through PyTorch: load, export_graph, to_mlir, backend

| Plan ID | Topic | Revision | Status |
|---|---|---|---|
| 2026-1029-002 | export-pytorch-exportedprogram | rev.27 (2026-11-04) | Implemented — Phases 0-5, with a C++ middle end (section 5.3b); SmolLM and ResNet-18 run on the PYNQ-Z1 NPU; the old path is removed |

Issue: #119 (software stack). Supersedes the front end built by
[2026-1029-001](2026-1029-001-export-load-quantize-to-mlir.md); its
compiler half is rewritten in C++ as `backend/`, one tool (`pynqo-backend`)
built against LLVM 22, and `src/vendors/npu_pynq/` is reorganised around it
(section 5.3d).

Goal: replace the hand-written export front end with open-source frameworks,
keeping as little custom code as possible:

```
load ──────────────────────> export_graph ─────────────────> to_mlir ────────────────────> backend
quantized nn.Module           ATen graph (ExportedProgram)    model.mlir (linalg)           package
transformers / torchvision    torch.export + functional       torch-mlir: FxImporter ->     partition by device:
+ our torchao configs         cache, default decompositions   torch dialect -> linalg       C++ pynqo-backend: cpu, npu
```

Quantization happens when the model is loaded: the framework applies our own
quantization configs, registered into torchao (`frontend/quantization/torchao/`), and
`load` then checks the result. When the backend's devices include the NPU, a
layer the NPU runs that is not quantized as the vendor's spec states is an
error, never passed on. Only a CPU-only build (the FP32 reference) accepts a
float model.

The current front end (`load/`, `quantize/`, `model_arch/`, `ir/`,
`pipeline.py`, `compilers/mlir/builder.py`, `compilers/mlir/ops/`) is not
kept alongside; it is removed once the new path passes its acceptance.

## Implementation status (2026-10-29)

Phase 0 answers, measured with torch 2.11, torchao 0.18, transformers 5.21
and iree-base-compiler 3.13; `to_mlir` has since moved to llvm/torch-mlir's
own wheels (rev.26, pinned in `src/export/requirements.txt`):

| Question | Answer |
|---|---|
| 1. Quantized contractions survive export | Yes. Our W8A8 module's `torch._int_mm` reaches linalg as an i8 x i8 -> i32 `linalg.matmul` on a constant weight; `--npu-claim` takes all 422 in SmolLM (211 per entry point). Decision 3.5 stays B. |
| 1b. transformers accepts our config | Not needed: `load` runs `quantize_(model, W8A8Config())` after `from_pretrained`, which is the same handler path. |
| 3. LLVM 22 parses the front end's output | Yes, once the output is upstream-only (see 5). |
| 3b. Weight form | `dense_resource` constants. Delegates read them for packing; a generic pass moves the rest into `weights.bin`. |
| 4. KV cache in place | The graph stays functional: the cache is an input and the updated cache an output, written with a select (`where`), not `index_put` (which lowers to `tm_tensor.scatter`, not upstream). The runtime keeps two cache sets and swaps them, so nothing is copied. |
| 5. Stop at linalg | `iree-turbine`'s `aot.export` and `iree-compile --compile-to=input` emit IREE-only ops (`util.global`, `hal.*`, `iree_linalg_ext.*`). Instead `to_mlir` uses torch-mlir as shipped in the IREE compiler wheel: its FX importer (`import_frozen_program`) and `iree-opt` running torch-mlir's two upstream pipelines. The result is linalg / tensor / arith / math only. |
| 6. Wheels | Python 3.12 on Windows and Linux. |

Other findings: transformers' `StaticCache` export recipe fails
`run_decompositions` with this torch; a functional cache class avoids
mutation altogether. `position_ids` must be passed explicitly (transformers
otherwise derives positions from the cache length).

Done:

- Front end: `frontend/load` (huggingface adapter, `check.py`),
  `frontend/quantization/torchao` (`W8A8Config`, `custom_quant/w8a8.py`:
  bit-exact with `src/test/model`), `frontend/export_graph` (decode and
  prefill with a functional cache), `frontend/to_mlir/torch_mlir`.
- Back end in C++ (`pynqo-backend`): partition with delegates and
  `__delegate__(id, in, out)`, generic weight externalization, CPU codegen
  in process (bufferization, out-params, LLVM IR, `llc`, link), the npu
  delegate adapter; vendor `custom_dialect/` (`npu.matmul`, `--npu-claim`,
  Pack, Encode).
- `driver/npu_rt/` in C++ (load, execute, PYNQ and SIM backends) and the
  runtime's C dispatcher (`src/runtime/delegate/`). Deviation from 5.3d: on
  the board the runtime allocates the NPU's buffers with `pynq.allocate` and
  passes them in (`npu_rt_requirements`, `npu_rt_platform`), so the driver
  does not depend on the image's allocator (libcma or XRT).
- `configs/smollm.yml`, `python -m src.export.cli build`, `src/deploy/smollm.py`.

Measured: a 2-layer SmolLM package on the simulator generates the same 16
greedy tokens as the exported PyTorch programs. The full model (30 layers,
W8A8, 164 MB weights) generates "A neural processing unit (NPU) is a type of
computer component that processes information in a way that mimics the way
the human brain processes information." on the simulator: 1,414,944 NPU jobs
for 32 tokens, the count recorded for the old package. The first 12 greedy
tokens equal the PyTorch reference; float rounding in CPU code then moves an
activation-quantization boundary.

Board (2026-11-01, PYNQ-Z1, g++ 11.2, CMake 3.22): the bundle from
`src/deploy/smollm.py` (165 MB) built the dispatcher and libnpu_rt on the
board in 33 s, then `src.runtime.cli run package --overlay npu_matrix.bit
--chat` generated the same 32 tokens as the host simulator, id for id, on
the physical NPU: 7,174 delegate calls, 1,414,944 jobs, 730,038,880 NPU
cycles, no ISA error. Prefill 47.2 s for 42 tokens, decode 0.51 tokens/s:
every call streams its segment's weights into the stage window, and q/k/v
and gate/up are separate contractions (7 per layer, not 4). Performance is
the next phase's work.

ResNet-18 (Phase 4, 2026-11-01): torchvision adapter, BatchNorm folded
before quantization, `W8A8Conv2d` (im2col plus one INT8 contraction),
`configs/resnet18.yml`. On the board the demo and the five gallery images are
classified correctly (Samoyed 89.9%), 21 NPU calls and 35,174 jobs per image
(the old package's count), about 6 s per inference.

Phase 5 (2026-11-01): the old front end (`load/`, `quantize/`, `ir/`,
`model_arch/`, `compilers/`, `compile.py`, `targets.py`), the vendor's
`compiler/` (`npu-opt`), `driver/npu_rt.c`, `abi.py`, `placement.py`,
`memory.py`, `isa.py`, `board_smoke.py`, the runtime's `engine.py`,
`package.py` and `cpu/`, the ONNX reference and `examples/smollm_onnx`
(decision 3.2: removed) are deleted with their tests and the export golden.
Moved: `isa/isa.py`, `isa/layout.py` -> `spec/`; `isa/sim.py` ->
`src/test/model/isa_sim.py`; `driver/dma.py` -> `driver/legacy/dma.py`;
`src/export/delivery/matrix.py` and the ResNet-18 asset downloader and pins
-> `src/deploy/`. CD builds both models through the new path; SmolLM's
board acceptance compares the board's 32 tokens with the host simulator's
run of the same package (`.github/cd/smollm.py reference`) instead of a
frozen ONNX Runtime output. `docs/rules/filetree.md` and `AGENTS.md` follow.

Not yet: performance (fused projections, resident weights, vectorized CPU
code), `aten/` kernels, W4A8, lit tests, macOS export.

---

## 1. Scope

In scope:

- **SmolLM2** (safetensors) and **ResNet-18** (TorchVision weights).
- `src/export/`; `src/vendors/npu_pynq/`, reorganised (section 5.3d);
  `src/deploy/`, moved out of `src/export/delivery/` (section 5.6);
  `src/test/`; and the CLI, CD, examples and docs that name these paths.

Deferred, as requested on 2026-10-29:

- **Qwen3** and **Qwen3.5**.
- **Qwen3-MoE** is deferred with them: it is Qwen3 attention plus routed
  experts, and its routing raises its own export question (section 7). This
  is an assumption to confirm.

Because the old front end is removed, the deferred models cannot be exported
between this change and their return; their packages already built keep
running on the board.

Out of scope:

- `src/runtime/` and the board, apart from: the KV-cache ABI (section 4,
  item 4); choosing a library by CPU model; and the delegate runtime
  (section 5.3c): `__delegate__`, a registry of device backends, the
  manifest's delegate table, and `npu_rt` loaded as the NPU backend. The board
  never needs PyTorch.
- `src/test/model/` numeric changes, unless decision 3.1 chooses B. One
  deliberate, numerics-free change is in scope: the ISA golden model moves
  in as `src/test/model/isa_sim.py` (section 5.3d).
- `src/jit/` (self-contained by rule).

## 2. Stages and frameworks

| Stage | Replaces | Framework |
|---|---|---|
| load | `load/` (safetensors reader, HF cache lookup, per-model `Config`), the ResNet checkpoint downloader, and `quantize/` (int8 numerics, weight plans, AWQ hook) | **transformers** `AutoModelForCausalLM` / `AutoTokenizer`, quantizing at load through `TorchAoConfig`; **torchvision** `get_model` / `get_model_weights` (downloads pretrained weights by name), then torchao `quantize_()`; **torchao** extended with our own configs (`AOBaseConfig` + `register_quantize_module_handler`, section 5.2) |
| export_graph | `model_arch/*` weight plans and per-model `to_mlir` | **torch.export** on the loaded (already quantized) model, the KV cache as a functional cache (an input, and the updated cache an output); quantized weights expand into int8 tensors and existing ATen ops such as `_int_mm`; `run_decompositions` with PyTorch's default table. Our own kernels for existing ATen ops are registered in C++ with **`TORCH_LIBRARY_IMPL`** (section 5.3) and built with **`torch.utils.cpp_extension`**. `decode` (one token) and `prefill` (`chunk` tokens) are separate programs with static shapes |
| to_mlir | `compilers/mlir/builder.py`, `compilers/mlir/ops/` (hand-written MLIR per op) | **torch-mlir**: its FX importer (`import_frozen_program`) takes the ExportedPrograms (one function per entry point) and emits the **torch dialect**; its two upstream pipelines convert that to **linalg**. torch-mlir comes from llvm/torch-mlir's dated release wheels and runs in process; IREE is not used |
| ResNet convolution | hand-written im2col | upstream **MLIR** transform `convert_conv2d_to_img2col` |
| backend | `compile.py`, `targets.py`, `compilers/mlir/toolchain.py`, `npu-opt` | **C++**: one tool, `pynqo-backend`, built on upstream **MLIR / LLVM 22** (section 5.3d). Delegates claim what they run and compile it (section 5.3c); the CPU module keeps one `__delegate__` call per claimed region and is lowered with upstream MLIR passes, `llc` and a linker |

## 3. Decisions to confirm before Phase 1

### 3.1 Numeric contract

`src/test/model/` is the numeric contract the RTL implements. torchao's
defaults differ from it (INT8 range possibly -128..127 instead of ±127, INT4
scale possibly amax/7.5 instead of amax/7, activation eps and granularity).

- **A (recommended):** the vendor's spec states the current contract, and
  `frontend/quantization/torchao/custom_quant/` follows it exactly (quant_min/max,
  eps, symmetric, per-token activations, per-channel and group-wise weights).
  No contract change.
- **B:** adopt torchao's defaults and change `src/test/model/` (and the
  RTL test vectors, if affected) deliberately. Requires explicit approval.

### 3.2 SmolLM ONNX path

It exists to keep the ONNX file's own INT8 quantization, so it cannot pass
through a PyTorch model.

- **Remove it (recommended)**, together with `compilers/onnx_mlir/` and the
  ONNX-only operators.
- Convert ONNX to PyTorch with onnx2torch (limited support for
  `MatMulInteger` and `DynamicQuantizeLinear`).
- Keep it as a separate path, which keeps `compilers/onnx_mlir/` and part of
  `compilers/mlir/ops/`.

### 3.3 AWQ

AWQ must run on the float model before it is quantized.

- **Offline (recommended):** an AWQ tool produces a checkpoint already
  quantized in our format, and `load` loads it like any other; the export
  flow never runs calibration. The tool must emit torchao tensors that
  `frontend/load/check.py` accepts (torchao's AWQ does), or `check.py` must learn the
  tool's format.
- **In the adapter:** `huggingface.py` calibrates and scales after loading,
  before quantizing; export becomes slower and the adapter grows.

### 3.4 SmolLM default

`configs/smollm.yml` follows the accepted build (W8A8 + AWQ, as recorded in
docs/goals/sw_optimal/Results.md). AWQ calibration needs transformers and
slows export; whether the default turns it on is open.

### 3.5 Quantized contraction: existing ATen ops or a new op

The NPU's quantized linear (INT8 weights, per-token INT8 activations) has no
single ATen op.

- **B (chosen, pending Phase 0):** use what torchao produces, existing ATen
  ops such as `_int_mm` plus the scale multiplications. No new op is defined;
  `frontend/export_graph/aten/` only supplies our own kernels for existing ATen ops
  (section 5.3), and `to_mlir/` needs no custom lowering.
- **A (fallback):** define a new op, `npu_export::quantized_linear`, with
  `TORCH_LIBRARY` (schema) and a Meta kernel (output shapes for
  `torch.export`), have the `custom_quant/` handlers make the quantized
  modules call it, and lower it in `frontend/to_mlir/custom.py`. Taken only if Phase 0 shows torchao's ops do not reach
  `pynqo-backend` in a form `--npu-claim` recognises.

### 3.6 Which schemes the NPU target accepts

- **W8A8 and W4A8 (recommended):** the multiply-accumulate is INT8 in both;
  W4A8 only stores weights in four bits, group-wise.
- **W8A8 only:** the simplest rule; the W4A8 path is dropped.

### 3.7 The FP32 reference build

A package's libraries are per CPU model (today named `host` and `pynq`), not
"with or without the NPU": today both call the NPU (the host library through the
C simulator backend in `npu_rt`, as `SharedExportEngineTest` checks). The FP32
reference is a CPU-only build instead:

- **`devices: [cpu]` (recommended):** no partition, no NPU programs, and
  `load` is not given a spec, so the float model is accepted. Phase 3's FP32
  acceptance runs through the same flow.
- **No FP32 build:** every export includes the NPU; the FP32 reference is
  produced outside the export flow.

### 3.8 Where `npu_rt` comes from (decided)

Decided by the delegate design (section 5.3c): `model.so` calls only
`__delegate__`, which the runtime provides; `libnpu_rt` is the runtime's NPU
backend, a C++ library built on the board and loaded by the runtime (section
5.3d). No model links hardware code, and `backend/cpu/` links no NPU file.
Today `npu_rt.c` is compiled into every `model.so`; that stops.

## 4. Custom code that remains

Only what the frameworks cannot do:

1. **The hardware spec** (`src/vendors/npu_pynq/spec/quant.py`): data only,
   the numeric formats the NPU accepts (section 5.2).
2. **Applying and checking it:** our quantization registered into torchao
   (`frontend/quantization/torchao/`: `config.py` and `custom_quant/`), applied by the
   `load` adapters, and `frontend/load/check.py`, which rejects a model the target
   cannot run.
3. **Our own kernels for existing ATen ops, in C++** (`frontend/export_graph/aten/`,
   section 5.3). They serve PyTorch's eager execution; the compiled path does
   not need them.
4. **The C++ backend, `pynqo-backend`** (section 5.3d), on upstream MLIR:
   - generic passes, written once for every delegate: outlining the claimed
     regions into one module per delegate and leaving
     `func.call @__delegate__(id, ...)`; and moving weight constants (torch-mlir
     emits weights as constants in the IR) into `weights.bin`, leaving views
     of the weight arena. The `[cpu]` build needs the second as well: the
     arena is the package's format, not the NPU's.
   - the npu delegate adapter, built on the vendor's passes: `--npu-claim`
     must recognise the INT8 contraction in the linalg torch-mlir produces (for
     example an `i8 x i8 -> i32` `linalg.matmul`) and the group-wise INT4
     form; `--npu-pack` writes the weights the NPU reads as tiles;
     `--npu-encode` writes ISA programs.
   - CPU codegen and the package (manifest with the delegate table).
   - KV cache in place: `torch.export` turns the cache update into "return a
     new cache". Copying the cache per token is too slow on the board, so the
     update must stay in place, through bufferization of the returned tensors
     into the state arguments or through a runtime ABI change. Decided in
     Phase 0.
5. **The delegate runtime:** `__delegate__` and the backend registry in
   `src/runtime/`, with `libnpu_rt` (`src/vendors/npu_pynq/driver/`, C++,
   built on the board) as the NPU backend (sections 5.3c, 5.3d).
6. **A thin pipeline, CLI and config loader:** chaining the stages, reading the
   YAML export configs (section 5.5), and stage files (`.pt2`, `.mlir`).

## 5. Layout after the change

`src/export/` is split at the hand-off file, `model.mlir` (linalg):

```
                front end: open frameworks, hardware-agnostic          back end: ours + vendor, by device
load ──────> export_graph ──────> to_mlir ──────> model.mlir (linalg) ──────> backend ──────> package
frontend/load/  frontend/export_graph/  frontend/to_mlir/                     backend/
```

- `frontend/` turns a PyTorch model into linalg with open frameworks
  (transformers, torchvision, torchao, torch.export, torch-mlir). It knows no
  hardware, apart from the quantization spec it is handed.
- `backend/` turns linalg into a package for the chosen devices, with our
  `pynqo-backend` (C++), the vendor's dialect and upstream MLIR / LLVM.
- The two never import each other; they meet only at `model.mlir`, which is
  also where the LLVM versions of torch-mlir and `pynqo-backend` meet (Phase 0). A new
  model framework changes only `frontend/`; a new accelerator changes only
  `backend/` and `src/vendors/`.

Each stage is one directory exposing one entry function from its
`__init__.py`:

| Stage | Directory | Entry function | Output | Framework |
|---|---|---|---|---|
| load | `frontend/load/` | `load(source=None, framework="auto", arch=None, weights=None, sha256=None, quantization=None, require=None) -> LoadedModel` | the model, quantized when `quantization` is given and checked against `require`; its tokenizer files or class metadata, source and config | transformers, torchvision, torchao |
| export_graph | `frontend/export_graph/` | `export_graph(module, shape) -> {"decode": ExportedProgram, "prefill": ExportedProgram}` | `decode.pt2`, `prefill.pt2`: Core ATen graphs (quantized when the model is) | torch.export |
| to_mlir | `frontend/to_mlir/` | `to_mlir(programs) -> Path` | `model.mlir` (linalg) | torch-mlir (FX importer, upstream pipelines) |
| backend | `backend/` | `pynqo-backend model.mlir --devices ... --cpus ... -o out` (Python: `build(mlir, devices, cpus, out)`) | the package directory | C++ on upstream MLIR / LLVM 22 |

```
src/export/
  pipeline.py            frontend (load -> export_graph -> to_mlir) -> backend, timing each stage
  cli.py                 argument parsing; config file + flag overrides; one subcommand per
                         stage (load, export-graph, to-mlir, backend) and export end to end
  configs/               one YAML export config per model or variant (section 5.5)
    smollm.yml  resnet18.yml

  frontend/              PyTorch -> MLIR (linalg); open frameworks only
    quantization/        shared library, not a stage; used by load/
      torchao/           our quantization, registered into torchao (section 5.2)
        __init__.py      importing it registers everything in custom_quant/ with torchao
        config.py        W8A8Config, W4A8Config (AOBaseConfig subclasses);
                         parameters from the spec pipeline.py passes in (spec/quant.py)
        custom_quant/    implementations, one file per scheme
          __init__.py    imports each scheme so its handler is registered
          w8a8.py        @register_quantize_module_handler(W8A8Config):
                         per-channel INT8 weights (+-127), per-token INT8 activations
          w4a8.py        @register_quantize_module_handler(W4A8Config):
                         group-wise INT4 weights ([-8, 7]), per-token INT8 activations

    load/                stage 1: PyTorch model, quantized at load
      __init__.py        load(source=None, framework="auto", arch=None, weights=None, sha256=None,
                         quantization=None, require=None) -> LoadedModel
      check.py           require(module, spec): every layer the NPU runs is quantized by one of
                         our configs as the spec states, else QuantizationError
      frameworks/        one generic adapter per framework; no model-specific code
        huggingface.py   any causal LM: AutoModelForCausalLM, AutoTokenizer, cache, dtype;
                         quantization through from_pretrained(quantization_config=TorchAoConfig(...))
        torchvision_models.py   any architecture: get_model(arch, weights=...) or a local
                         checkpoint, then torchao quantize_() when quantization is given;
                         categories and preprocessing from the weights' metadata

    export_graph/        stage 2: ATen graph (ExportedProgram per entry point)
      __init__.py        export_graph(module, shape): torch.export -> default decompositions
      export.py          torch.export with a functional KV cache (decode, prefill)
      aten/              our own C++ kernels for existing ATen ops (section 5.3)
        __init__.py      load_kernels(): build if stale (torch.utils.cpp_extension) and load
        op_register.cpp  TORCH_LIBRARY_IMPL(aten, CPU, m) for cpu/; later
                         TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) + CPU fallback for npu/
        cpu/<op>.cpp     kernel bodies for CPU
        npu/<op>.cpp     reserved: kernel bodies for the NPU device
        device_register.cpp  added with npu/: the "npu" PyTorch device (name, allocator,
                         guard, empty / copy kernels)

    to_mlir/             stage 3: MLIR (linalg)
      __init__.py        to_mlir(programs) -> torch.mlir: to_torch_dialect
      torch_mlir/        one file per dialect step (llvm/torch-mlir's Python package)
        __init__.py
        to_torch_dialect.py   FxImporter.import_frozen_program -> torch.mlir (torch dialect)

  middle-end/            stage 3b: torch.mlir -> model.mlir (linalg + tensor + arith + math); C++
    pynqo_middle_end.cpp   our rewrites, then torch-mlir's two upstream pipelines
    patch_torch_mlir.cmake    builds torch-mlir without Python bindings or its tests
    torch/transforms/    one file per rewrite on the torch dialect (section 5.3b)

  backend/               stage 4: MLIR (linalg) -> package; C++ only, no Python (section 5.3d)
    CMakeLists.txt       builds pynqo-backend against LLVM 22 (MLIR); links custom_dialect
    pynqo_backend.cpp      the tool: model.mlir + options -> package (partition -> (cpu ||
                         each delegate) -> package); also runs single passes for debugging
    partition.cpp/.h     each delegate claims; claimed regions are outlined, one module per
                         delegate, with func.call @__delegate__(id, ...) left in cpu.mlir;
                         weight constants -> weights.bin + views of the weight arena
    device_register.cpp/.h   cpu (always) and the registered delegates, [cpu] or [cpu, npu]
                         only; CPU models named <arch>-<model> (arm-cortex-a9, the export
                         machine's own model)
    cpu/                 upstream MLIR / LLVM only; knows no delegate
      toolchain.cpp/.h   build(cpu.mlir, cpus): rejects any op outside upstream dialects,
                         then per CPU model: upstream lowering to LLVM IR, llc, link
                         -> one shared library each (LLVM 22)
    delegates/
      delegate.h         the Delegate interface: claims(), compile()
      npu/
        lower.cpp/.h     the npu delegate, an adapter over custom_dialect: claims through
                         --npu-claim; compiles through --npu-pack (tiles) and --npu-encode
                         (ISA programs)
    package.cpp/.h       manifest.json (with the delegate table) and the packaged files
    test/                lit + FileCheck tests, one per pass

src/deploy/              not part of the export flow: board deployment bundles (section 5.6)
  resnet18.py            from src/export/delivery/resnet18.py
  resnet18_assets.py     sample image and gallery download (from model_arch/resnet18/download.py)
  resnet18-*.json        pins for those demo images (from model_arch/resnet18/)
  matrix.py              from src/export/delivery/matrix.py

src/vendors/npu_pynq/           reorganised (section 5.3d)
  spec/                  hardware facts, data only, imports nothing; shipped to the board
    __init__.py
    isa.py               instruction encoding
    layout.py            how a tiled GEMM sits in DDR
    quant.py             quantization formats the NPU accepts: W8A8, W4A8
    board.py             the board's CPU: Cortex-A9, ARMv7 hard-float, NEON
  custom_dialect/        compile time, C++, never shipped; a library pynqo-backend links
    CMakeLists.txt
    include/Npu/
      NpuOps.td          the dialect: npu.weight, npu.matmul
      NpuDialect.h
      NpuPasses.h        registers --npu-claim, --npu-pack, --npu-encode
      NpuEncoder.h       C++ port of spec/isa.py + spec/layout.py
    lib/
      NpuDialect.cpp     op verifiers
      Claim.cpp          --npu-claim: NPU contractions -> npu.matmul / npu.weight, marked
      Pack.cpp           --npu-pack: the weights the NPU reads -> B tiles in weights.bin
      Encode.cpp         --npu-encode: claimed regions -> programs.bin + programs.json
      NpuEncoder.cpp
    test/                lit + FileCheck, one per pass
  driver/                run time; C++ built on the board (section 5.3d)
    npu_rt/              libnpu_rt.so: the NPU delegate backend __delegate__ reaches
      CMakeLists.txt     built on the board (PYNQ's g++); natively on the export machine for SIM
      npu_rt.h           extern "C": npu_rt_load(package_dir, mmio_base), npu_rt_execute(id,
                         buffers), npu_rt_close()
      load.cpp           read programs.json / weights.bin, allocate CMA (libcma) or host
                         memory, relocate the ISA programs, check the overlay geometry
      execute.cpp        per __delegate__ call: pack the INT8 input into A tiles, run the
                         program, unpack the INT32 partial planes
      pynq.cpp           the board backend: /dev/mem MMIO, PROG_ADDR / START, poll
      sim.cpp            the export machine's backend: C++ port of the ISA golden model
    overlay/             the FPGA bitstream's manifest (Python, unchanged)
      __init__.py
      build.py           after Vivado: write the BIT/HWH provenance manifest
      verify.py          before loading and before a release: check BIT/HWH against it
    legacy/
      dma.py             ABI 1.0 DMA matrix runtime (matrix demo and src/deploy/matrix.py only)
```

`frontend/export_graph/__init__.py` exports the model it is given:

```python
def export_graph(module, shape):
    programs = export.export(module, shape)                # {"decode": EP, "prefill": EP}
    return {name: ep.run_decompositions(torch.export.default_decompositions())
            for name, ep in programs.items()}
```

Imports point forward only: a stage never imports a later one, and
`frontend/` never reaches `backend/` except by running `pynqo-backend`. The quantization spec
reaches `frontend/load/` as an argument from `pipeline.py`, which reads it
from `src/vendors/npu_pynq/spec/quant.py` when the devices include `npu`. `frontend/load/` imports
`frontend/quantization/`; `frontend/export_graph/` does not.
`backend/` (C++) links the vendor's `custom_dialect/` library through its
npu delegate adapter; `src/vendors/` never includes or imports `src/export/`.
`test_export_layering.py` and `test_vendor_boundary.py` check these edges.

Removed: the current `load/`, `quantize/`, `model_arch/`, `ir/`,
`pipeline.py`, `compile.py`, `targets.py` and `compilers/` (its
`mlir/toolchain.py` becomes `backend/cpu/toolchain.cpp`; `mlir/builder.py`
and `mlir/ops/` are replaced by torch-mlir; `onnx_mlir/` goes with decision 3.2), and
`npu-opt` (replaced by `pynqo-backend`). Expected size of `src/export/`'s
Python: about 800-1200 lines (from about 5700), plus the C++ backend and the
vendor changes.

### 5.1 `load` takes parameters, not model names

`load` knows no model names, and its framework adapters hold no
model-specific code: which model to load is entirely the config's choice.
With `framework="auto"` the framework is detected from the parameters:

| Parameters | Framework | Generic API |
|---|---|---|
| `source` is a Hugging Face repo id, or a directory holding `config.json` | `huggingface` | `AutoModelForCausalLM`, `AutoTokenizer` (the class comes from `config.json` `architectures`) |
| `arch` + `weights` (a TorchVision weights name, e.g. `IMAGENET1K_V1`) | `torchvision` | `get_model(arch, weights=get_model_weights(arch)[weights])`, which downloads and checks the weights |
| `arch` + `source` (a local `.pth` checkpoint) | `torchvision` | `get_model(arch)` + `load_state_dict` |

The returned model is on the CPU and in `eval()` mode: float32 (the FP32
reference) when `quantization` is not given, quantized by our configs when it
is (section 5.2). `sha256`, when given, is checked against the file actually read
(TorchVision itself only checks the hash prefix in the file name). `load`
returns a `LoadedModel`: the `nn.Module`, the files to copy into the package
(tokenizer), metadata the package needs (TorchVision categories and
preprocessing from the weights' `meta` and `transforms()`), the source and the
config, so later stages never ask which framework produced the model.

A new model on an existing framework needs only a config file. A new
framework is one adapter in `frontend/load/frameworks/`. A model-specific module is
added only when a framework's generic API cannot load the model (weight
renaming, or an architecture the framework lacks). Adapter file names avoid
the names of the packages they wrap (`huggingface.py`, not `transformers.py`)
so they never shadow them.

ResNet-18 therefore needs no directory: the checkpoint comes from TorchVision
(the old model pin's full SHA-256 moves into `configs/resnet18.yml`), the
class names from the weights' metadata, and the demo sample image and gallery,
which are example assets rather than export inputs, move to `src/deploy/`
with their pins. `model_arch/resnet18/requirements.txt` merges into the one export
requirements file.

### 5.2 Quantization at load, registered into torchao

A PyTorch model's code defines no quantization: `LlamaForCausalLM` and
TorchVision's `ResNet` hold plain `nn.Linear` and `nn.Conv2d`. torchao
attaches it afterwards, by replacing a module's `weight` with a quantized
tensor (int8 data, scale, and how activations are quantized at run time); the
module class stays `nn.Linear`. transformers calls torchao itself when given a
`TorchAoConfig`; for TorchVision the adapter calls `torchao.quantize_()`.
Either way torchao only sees an `nn.Module`, so quantization is the same for
every framework.

Hardware toolchains let the target decide quantization (ExecuTorch backends
ship their own quantizers; Edge TPU and Arm Ethos-U refuse models that are not
INT8). This plan splits it by layer:

- **The vendor states hardware facts, as data.**
  `src/vendors/npu_pynq/spec/quant.py` lists the schemes the NPU accepts
  and their exact numerics: W8A8 (per-output-channel INT8 weights within
  ±127, per-token INT8 activations) and W4A8 (group-wise INT4 weights in
  [-8, 7], one scale per group and output channel, INT8 activations), as
  defined in `src/test/model/` (decision 3.1; 3.6 for which schemes). It holds
  no PyTorch code.
- **`frontend/quantization/torchao/` implements them, registered into torchao** without
  modifying torchao, the way `aten/` registers kernels into ATen:

  | | `frontend/quantization/torchao/` | `frontend/export_graph/aten/` |
  |---|---|---|
  | registered into | torchao | ATen |
  | the list | `config.py`: our configs | `op_register.cpp`: op -> kernel |
  | implementations | `custom_quant/<scheme>.py` | `cpu/<op>.cpp`, `npu/<op>.cpp` |
  | mechanism | `@register_quantize_module_handler(Config)` | `TORCH_LIBRARY_IMPL(aten, <key>, m)` |

  `config.py` holds `AOBaseConfig` subclasses (`W8A8Config`,
  `W4A8Config`) whose parameters come from the spec. Each file in
  `custom_quant/` registers the handler `quantize_()` calls for every
  `nn.Linear`: it computes the INT8 (or INT4) weights and scales exactly as
  `src/test/model` does, wraps them in torchao's quantized tensor together
  with the per-token activation rule, and puts the result back as the
  module's weight. Usage is the same as torchao's own configs:
  `quantize_(model, W8A8Config())`, or
  `TorchAoConfig(quant_type=W8A8Config())` for transformers (accepting a
  config object there depends on the transformers version; checked in
  Phase 0). An own tensor subclass is added only if Phase 1 shows torchao's
  quantized tensor cannot express the contract.
- **`load` applies and checks them.** `quantization` names the config to
  apply; `require` is the target's spec. `frontend/load/check.py` then requires every
  layer the NPU runs to be quantized by one of our configs with the spec's
  parameters, and raises otherwise:

  ```
  QuantizationError: the pynq target requires W8A8 (src/vendors/npu_pynq/spec/quant.py)
    model.layers.3.mlp.down_proj    float32 weights
    model.layers.7.self_attn.o_proj int8 weights, asymmetric (zero point != 0)
  ```

  Checking only "is it 8-bit" is not enough: an 8-bit model quantized another
  way (asymmetric, another range or granularity) computes wrong results on the
  NPU, so it is rejected too. The first version accepts only our torchao
  configs; other formats (GPTQ or AWQ packages' own modules) are rejected with
  a message saying so.
- **The devices select the spec.** When the backend's devices include `npu`,
  `pipeline.py` reads the NPU's spec (`src/vendors/npu_pynq/spec/quant.py`) and
  passes it to `load` as `require`; with `[cpu]` alone (the FP32 reference,
  decision 3.7) nothing is required, so a float model is accepted and
  everything runs on the CPU. `--npu-claim` takes only quantized
  contractions, so FP32 ones stay on the CPU.
- **Boundary rules are unchanged:** `src/vendors/` never imports `src/export/`
  (`test_vendor_boundary.py`). The spec needs neither torch nor `src/export`;
  the board package ships `spec/` and the `driver/` sources only, so the board still never
  needs PyTorch.

torchao states that it is minimizing new feature development and moving
quantization into `torch` and the frameworks that use it, so its version is
pinned and everything torchao-specific stays in `frontend/quantization/torchao/` and
`frontend/load/check.py`.

### 5.3 Our own kernels for existing ATen ops (C++)

ATen already defines the ops; `frontend/export_graph/aten/` registers our own kernels
for some of them, without modifying or rebuilding PyTorch. The dispatcher
keeps a table of (op, dispatch key) -> kernel at run time;
`TORCH_LIBRARY_IMPL` adds our entries when our library is loaded into the
Python process. Headers and libraries come from the installed torch wheel; no
ATen source is downloaded.

Modifying PyTorch's source (`native_functions.yaml`, in-tree kernels, a new
built-in dispatch key) is needed only to add an op to PyTorch itself, change
an op's schema or contribute upstream. Older out-of-tree backends forked
PyTorch because the PrivateUse1 mechanism did not exist yet.

```
frontend/export_graph/aten/
  __init__.py          load_kernels(): build if stale and load the library
  op_register.cpp      which op uses which kernel, at a glance
  cpu/<op>.cpp         kernel bodies for CPU
  npu/<op>.cpp         reserved: kernel bodies for the NPU device
  device_register.cpp  added with npu/: the "npu" device
```

```cpp
// op_register.cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("mm", TORCH_FN(cpu::mm));                      // cpu/mm.cpp
}

// added with npu/: tensors on device "npu" dispatch here
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", TORCH_FN(npu::mm));                      // npu/mm.cpp
}
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {                 // ops without an NPU kernel run on the CPU
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu::cpu_fallback>());
}
```

A new kernel is one file in `cpu/` or `npu/` plus one line in
`op_register.cpp`. Several `TORCH_LIBRARY_IMPL` blocks may register to the
same (namespace, dispatch key), so `device_register.cpp` and
`op_register.cpp` each keep their own.

- **CPU kernels (`cpu/`)** replace PyTorch's own CPU kernel for that op in the
  whole process once loaded (PyTorch warns about the override, and the
  original kernel can no longer be called as a fallback). They need neither a
  device registration nor a Meta kernel: CPU is a built-in device, and
  existing ATen ops already have Meta kernels for `torch.export`. Where only
  part of a program should use them, a `TorchDispatchMode` scopes the
  replacement to a `with` block instead.
- **NPU kernels (`npu/`)** run when a tensor lives on device `npu`, the name
  given to PyTorch's PrivateUse1 backend. `device_register.cpp` then makes
  the device usable, in three layers:

  | Layer | What | Where |
  |---|---|---|
  | name | `c10::register_privateuse1_backend("npu")` (or `torch.utils.rename_privateuse1_backend`) | `device_register.cpp` |
  | infrastructure | allocator, device guard (`C10_REGISTER_GUARD_IMPL`), and the kernels for `empty.memory_format`, `empty_strided` and `_copy_from` | `device_register.cpp` |
  | computation | the ops in `npu/`, with a CPU fallback for the rest | `op_register.cpp` |

  On the Zynq the CPU and the NPU share DDR, so the device can live in host
  memory: the CPU allocator (or CMA buffers from the vendor driver, if the NPU
  reads them by DMA), `memcpy` for copies and an almost empty guard. NPU
  kernels call into `src/vendors/npu_pynq/driver/` (the board or `npu_rt`'s
  C simulator backend), so
  hardware details stay in the vendor. `npu/` and `device_register.cpp` are
  created with the first NPU kernel.
- **Scope:** these kernels run only when PyTorch executes eagerly (host
  reference runs, calibration, a future on-board PyTorch). `torch.export`
  records op names such as `aten.mm`, not kernels, and the package's code is
  generated from MLIR by `pynqo-backend` and LLVM. A kernel meant to run inside the
  compiled package needs a matching rule at the lowering level as well; that
  is planned separately once such a kernel exists.
- **Build:** `torch.utils.cpp_extension.load` compiles on first use and when a
  source changes, into the git-ignored `build/export-graph-aten/`, as `pynqo-backend`
  builds itself; loading the library runs the registrations. CMake with
  `find_package(Torch)` is the alternative. LibTorch's stable ABI (shown in
  the PyTorch extension example) is considered so the library need not be
  rebuilt for every torch version.
- **Tests:** each kernel matches PyTorch's own kernel (or `src/test/model`
  where the numeric contract applies); `torch._C._dispatch_dump("aten::<op>")`
  confirms our entry is registered; tests skip where no compiler or torch is
  available.

References:
<https://docs.pytorch.org/tutorials/advanced/privateuseone.html>,
<https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html>,
<https://docs.pytorch.org/docs/stable/library.html>,
<https://docs.pytorch.org/docs/stable/cpp_extension.html>,
<https://github.com/pytorch/extension-cpp>,
<https://github.com/pytorch/vision/tree/main/torchvision/csrc/ops>.

### 5.3b `to_mlir` with torch-mlir

ATen reaches MLIR through a ladder of dialects; each framework step owns one
rung, so `frontend/to_mlir/torch_mlir/` has one file per step:

```
ExportedProgram (aten.mm, aten._int_mm, ...)
   │  to_torch_dialect.py: FxImporter.import_frozen_program, one op per ATen op
   ▼
torch dialect (torch.aten.mm, ...)                          -> torch.mlir
   │  middle-end/ (C++): torch/transforms/, then
   │  torch-function-to-torch-backend-pipeline and
   │  torch-backend-to-linalg-on-tensors-backend-pipeline
   ▼
linalg + tensor + arith + math    -> model.mlir  ->  backend/ (pynqo-backend, C++)
```

The importer takes one ExportedProgram per call into the same module, so
`decode` and `prefill` become two functions of one module, and a weight both
use is one `dense_resource` constant. Nothing in `to_mlir/` writes MLIR by
hand: `compilers/mlir/builder.py` and `compilers/mlir/ops/` are deleted.

`middle-end/` is `pynqo-middle-end`, a C++ tool with `OpRewritePattern` passes on
the torch dialect, where layers are still recognisable. The torch-mlir wheels hold
no C++ API and bind another LLVM, so it builds torch-mlir from source as a CMake
dependency, against the LLVM 22 `pynqo-backend` uses, without building LLVM. That
pins torch-mlir at `ac33bab` (2026-01-14), whose LLVM is closest to the 22.x
branch; what later versions added and our models need is a rewrite here:
`DecomposeIntMm` (`_int_mm` -> `mm`) and `AnyDimToAnyDims`. The importer stays the
newer wheel in Python, so `torch.mlir` crosses versions as text.

`FuseConvBatchNorm` folds BatchNorm with the arithmetic of PyTorch's
`fuse_conv_bn_weights`; torch-mlir cannot lower BatchNorm itself
(`_native_batch_norm_legit_no_training` arrives as a generic `torch.operator`),
so every model with BatchNorm needs it. In a float model the convolution's
weight and bias take the BatchNorm. In a W8A8 model `W8A8Conv2d` has already
become `_int_mm` plus a rescale, so `w_scale` and the bias take it and the INT8
weights the NPU reads stay as quantized from the unfolded weight. A channel
whose BatchNorm multiplier is negative keeps its weights and gets a negative
`w_scale`. This replaced folding before quantization (`fold_batchnorm`, removed
2026-11-04). The results are the same up to rounding, not bit for bit: activation
quantization turns a last-bit float difference into a flipped INT8 value, so
ResNet-18's probabilities move slightly (Samoyed 89.87% -> 90.20%).

torch-mlir comes from its own project, <https://github.com/llvm/torch-mlir>:
the dated wheels of `llvm/torch-mlir-release` (no release on PyPI), pinned per
platform in `requirements.txt`, which names their index with `--find-links`.
Its Python package holds the FX importer. `torch.mlir` is written beside
`model.mlir` and is what `pynqo-middle-end` reads.

IREE is not used. Its input path (`iree-turbine` `aot.export`,
`iree-compile --compile-to=input`) emits IREE-only ops (`util.global`,
`hal.*`, `iree_linalg_ext.*`) that `pynqo-backend` does not parse; an earlier
revision took torch-mlir from the IREE compiler wheel (`iree-opt`), which
rev.26 replaced. A lowering that leaves a non-upstream op (for example
`index_put` -> `tm_tensor.scatter`) is avoided in the model, not handled in
the back end.

### 5.3c `backend`: delegates

Partition follows the delegate pattern used by ExecuTorch, TensorFlow Lite,
ONNX Runtime execution providers and TVM's BYOC: each device backend says
what it can run, the claimed parts become that backend's data, the host code
keeps one device-agnostic call per claimed part, and the device's runtime
library belongs to the runtime, not to the model.

```
model.mlir (linalg)
  │  partition.cpp       each delegate claims ops (npu: --npu-claim); claimed regions
  │                      are outlined, one module per delegate; weight constants move
  │                      to weights.bin
  ├─> cpu.mlir           upstream dialects only; each claimed region is now
  │   │                  func.call @__delegate__(id, buffers...)
  │   └─> cpu/toolchain.cpp  upstream lowering -> LLVM IR -> llc -> link
  │                          -> model.so per CPU model (no hardware code linked)
  ├─> npu.mlir           the regions the npu delegate claimed
  │   └─> delegates/npu/lower.cpp   --npu-pack (tiles into weights.bin),
  │                                 --npu-encode -> programs/ (ISA programs)
  └─> (another delegate's module, compiled by that delegate)
  │
  package.cpp            model.so + programs/ + weights.bin + manifest.json
                         (with the delegate table) + tokenizer -> package
```

Before and after partition:

```mlir
// model.mlir
%y = linalg.matmul ins(%xq, %w : tensor<16x576xi8>, ...) outs(...)    // the NPU can take this

// cpu.mlir
func.func private @__delegate__(i64, memref<?xi8>, memref<?xi8>, memref<?xi32>)  // from the runtime
func.call @__delegate__(%c3, %xq_buf, %arena, %y_buf) : (...) -> ()              // delegate #3
```

```json
"delegates": {"3": {"device": "npu", "program": "programs/3.bin", "inputs": ["..."], "outputs": ["..."]}}
```

- **Why the CPU module calls a delegate at all.** The CPU runs the model from
  start to end (`decode`, `prefill`); at each claimed region it must hand its
  inputs over, run the region elsewhere and take the result back.
  `func.call @__delegate__(id, ...)` is that hand-over, and it names no
  device: `model.so` does not know whether delegate 3 is the NPU.
- **Compile side: a delegate answers two questions** (`delegates/delegate.h`):

  | Interface | npu delegate (`delegates/npu/lower.cpp`) |
  |---|---|
  | `claims(module)`: which ops it takes | `--npu-claim`: INT8 contraction, INT4 groups |
  | `compile(module)`: what the claimed part becomes | `--npu-pack` (tiles), `--npu-encode` -> ISA programs |

  What a device requires from the front end (the quantization spec) is not
  part of the C++ interface: only the front end uses it, so `pipeline.py`
  reads it from the vendor's `spec/quant.py`.

  Claiming is device knowledge and stays in the vendor's `--npu-claim`.
  Outlining claimed regions and writing the `__delegate__` calls is
  device-agnostic, in `backend/partition.cpp`. `npu.call`,
  `npu.module` and `--npu-lower-host` are deleted.
- **Run side: the runtime provides `__delegate__`.** It looks the id up in the
  manifest's delegate table and calls the backend registered for that
  device (an `init` / `execute` pair, as ExecuTorch backends register).
  The NPU backend is `npu_rt` (`src/vendors/npu_pynq/driver/`), built as its
  own library and loaded by the runtime (decision 3.8). On the export
  machine the same call reaches `npu_rt`'s SIM backend, a C port of the ISA
  golden model.
- **The `__delegate__` ABI** is fixed before any delegate is written, because
  every `model.so` and every backend must follow it: how buffers are passed
  (a fixed list of memref descriptors, or a pointer array and a count), how
  outputs come back, and how errors are reported. Defined and tested in
  Phase 2.
- **Granularity.** The interface allows a delegate to claim a whole
  subgraph; the first npu delegate claims one contraction per call, as the
  NPU runs today.
- **A new device** is one directory in `backend/delegates/` (and its
  vendor's passes) plus one backend registered in the runtime; the generic
  passes, CPU codegen and the `__delegate__` calls in `model.so` do not
  change.
- **Packing and offsets.** `cpu.mlir` reads its own weights at offsets in the
  weight arena, and the npu delegate stores the weights it reads as tiles;
  both layouts are fixed in one packing step that runs once partition has
  decided which weights each side reads.

- **Devices, not targets.** `device_register.cpp` holds `cpu`, which is always
  present, and the registered delegates, today only `npu`. It accepts `[cpu]`
  (the FP32 reference) and `[cpu, npu]`; anything else, `[npu]` alone
  included, is an error: the CPU always runs what no delegate claims.
- **CPU models** (`device_register.cpp`) are named `<arch>-<model>`, each with what LLVM
  needs (as `Target` holds today: triple, CPU, features, float ABI, index
  width, library file name). The board's facts are data in
  `src/vendors/npu_pynq/spec/board.py`; the thin Python layer passes them to
  `pynqo-backend` as options, so the C++ back end reads no Python and no
  vendor source:

  ```
  pynqo-backend model.mlir --devices=cpu,npu \
      --cpu=arm-cortex-a9:triple=armv7a-unknown-linux-gnueabihf,cpu=cortex-a9,features=+neon,+vfp3,float-abi=hard \
      --cpu=native -o build/pkg/smollm       # native: detected, e.g. x86_64-alderlake
  ```

  | CPU model | Source | Runs |
  |---|---|---|
  | `arm-cortex-a9` | `src/vendors/npu_pynq/spec/board.py` (ARMv7 hard-float, NEON) | on the board, NPU tasks on the NPU |
  | the export machine's model, e.g. `x86_64-alderlake`, `x86_64-znver4`, `aarch64-apple-m2` | detected through LLVM (its host CPU name) | on that machine, NPU tasks through `npu_rt`'s SIM backend |

  A config may write `native` for the export machine; the package records the
  detected model. The export machine's library is **named** after its model
  but **compiled for its architecture's generic baseline** (`-mcpu` generic,
  a Linux `model.so` or a macOS `model.dylib`), so a package still runs on an
  older machine of the same architecture.
- **The runtime picks a library by role, not by name.** Today
  `src/runtime/core/package.py` looks up `manifest["libraries"]["pynq"]` or
  `["host"]`. With models as names, each library entry in the manifest also
  records where it runs (`board` or `simulator`), and the runtime selects by
  that field. Older packages, keyed `host` / `pynq`, keep loading.
- **NPU-specific work lives in `backend/delegates/npu/`** over the
  vendor's `custom_dialect/` passes: which contractions the NPU takes, weight
  packing, ISA encoding. `partition.cpp` and `cpu/` know no
  device.
- **The CPU path uses upstream MLIR / LLVM, not IREE.** IREE's CPU back end
  emits a `.vmfb` run by the IREE runtime; the board's runtime loads a plain
  `model.so` (C ABI, `decode` / `prefill`), so this path stays as today.
- The back end's delegate registry is unrelated to
  `frontend/export_graph/aten/device_register.cpp`, which registers a PyTorch
  device for eager execution.

### 5.3d The back end in C++, and `src/vendors/npu_pynq/`

Frameworks that compile for accelerators keep everything that manipulates IR
in C++ and use Python as a thin layer: IREE (MLIR passes and `iree-compile`),
TVM (a C++ core behind its Python API), ONNX Runtime and TensorFlow Lite
(C++ partitioning and delegates). The front end stays in Python because it
lives next to PyTorch; the back end becomes one C++ tool, `pynqo-backend`. Only
the language changes: `backend/` keeps the structure agreed for it
(`partition`, `device_register`, `cpu/`, `delegates/<device>/`, `package`),
each Python module becoming a C++ source with its header beside it, plus
`pynqo_backend.cpp` (the tool's entry) and `test/`.

- **All passes run in one process**: claim, outline, externalize weights,
  pack, encode and the lowering to LLVM, with no MLIR text written and read
  back between tools.
- **Delegates register in C++** (`REGISTER_DELEGATE`), as MLIR registers
  dialects and passes.
- **Tests follow upstream MLIR**: `lit` + `FileCheck` per pass (input IR,
  checked output IR), both shipped with LLVM.
- **CPU codegen, first version**: upstream lowering to the LLVM dialect and
  translation to LLVM IR in process, then `llc` and the linker as today;
  emitting objects through LLVM's `TargetMachine` directly comes later.
- `pynqo-backend` also runs single passes, opt-style, for debugging; it
  replaces `npu-opt`.
- **No Python in `backend/`.** `pipeline.py` runs `pynqo-backend` with the
  config as options (the board's CPU facts read from `spec/board.py`), and
  builds it first when a source is newer than the binary, as `npu-opt` builds
  itself on first use today, so the guides that rely on that keep working.
  Python never imports `backend/`; the two meet at `model.mlir` and the
  command line.

| Python (agreed) | C++ |
|---|---|
| `backend/__init__.py` `build()` | `pynqo_backend.cpp` (the pipeline); building and running it moves to `pipeline.py` |
| `partition.py` | `partition.cpp/.h` (outline + `__delegate__` calls, weight constants into the arena) |
| `device_register.py` | `device_register.cpp/.h` |
| `cpu/__init__.py`, `cpu/toolchain.py` | `cpu/toolchain.cpp/.h` |
| `delegates/__init__.py` | `delegates/delegate.h` |
| `delegates/npu/__init__.py`, `lower.py` | `delegates/npu/lower.cpp/.h` |
| `package.py` | `package.cpp/.h` |
| — | `test/` (`lit` + `FileCheck`) |

**The vendor never depends on the back end.** `delegates/delegate.h` lives in
`src/export/backend/`; if the vendor implemented it, the vendor would include
`src/export`. Instead the vendor ships its dialect and passes as a library
that knows nothing of delegates, and `backend/delegates/npu/` wraps
those passes into the npu delegate, as MLIR tools combine independent dialect
libraries.

`src/vendors/npu_pynq/` after the change:

```
src/vendors/npu_pynq/                      where each file comes from
  spec/
    __init__.py                              from isa/__init__.py
    isa.py                                   from isa/isa.py
    layout.py                                from isa/layout.py
    quant.py                                 new
    board.py                                 from compiler/target.py
  custom_dialect/                            from compiler/mlir/
    CMakeLists.txt                           from compiler/mlir/CMakeLists.txt
    include/Npu/
      NpuOps.td                              npu.call and npu.module removed
      NpuDialect.h
      NpuPasses.h                            registerHostLowerPass removed
      NpuEncoder.h
    lib/
      NpuDialect.cpp                         CallOp verifier removed
      Claim.cpp                              from lib/Partition.cpp, claiming only
      Pack.cpp                               from compiler/package.py (Python -> C++)
      Encode.cpp                             EncodePass from lib/Lower.cpp
      NpuEncoder.cpp                         from lib/NpuEncoder.cpp
    test/                                    new
  driver/
    npu_rt/
      CMakeLists.txt                         new
      npu_rt.h                               new: the C interface
      load.cpp                               from driver/placement.py + memory.py (Python -> C++)
      execute.cpp                            from driver/npu_rt.c (pack, run, unpack)
      pynq.cpp                               from driver/npu_rt.c pynq_run + driver/isa.py's checks
      sim.cpp                                from driver/npu_rt.c sim_run
    overlay/
      __init__.py                            unchanged
      build.py                               unchanged
      verify.py                              unchanged
    legacy/
      dma.py                                 from driver/dma.py

moved out: isa/sim.py -> src/test/model/isa_sim.py (the ISA golden model)
deleted: compiler/passes.py, compiler/package.py (now Pack.cpp), lib/Lower.cpp's
HostLowerPass, compiler/mlir/tools/npu-opt.cpp (pynqo-backend replaces it);
driver/abi.py (no Python struct mirrors left), driver/isa.py's IsaDevice.run, gemm
and NpuGemm (they duplicate npu_rt's board backend), driver/placement.py and
driver/memory.py (now load.cpp), driver/__init__.py;
driver/board_smoke.py is a separate decision
```

**`custom_dialect/`: compile time, C++, never shipped.** A library that
`pynqo-backend` links; `backend/delegates/npu/lower.cpp` calls its three passes.

```
custom_dialect/
  CMakeLists.txt         TableGen + the passes -> a static library (LLVM 22 / MLIR)
  include/Npu/
    NpuOps.td            the dialect: npu.weight (a named weight the NPU reads),
                         npu.matmul (an INT8 x INT8 -> INT32 contraction on the NPU)
    NpuDialect.h         generated op classes and the dialect
    NpuPasses.h          registers --npu-claim, --npu-pack, --npu-encode
    NpuEncoder.h         C++ port of spec/isa.py + spec/layout.py: ISA words, tile plans
  lib/
    NpuDialect.cpp       op verifiers (static shapes, K-chunk planes, rhs from npu.weight)
    Claim.cpp            --npu-claim            (from Partition.cpp, claiming only)
    Pack.cpp             --npu-pack             (from compiler/package.py)
    Encode.cpp           --npu-encode           (EncodePass from Lower.cpp)
    NpuEncoder.cpp       the encoder behind Pack and Encode
  test/                  lit + FileCheck: one input and expected output per pass
```

| Pass | Reads | Writes | Called by `lower.cpp` as |
|---|---|---|---|
| `--npu-claim` | `model.mlir` (linalg) | the same module, with every INT8 contraction (and INT4 group form) the NPU can run rewritten to `npu.matmul` / `npu.weight` and marked for the npu delegate | `claims()` |
| `--npu-pack` | `npu.mlir` (the claimed regions) | the weights `npu.matmul` reads, as B tiles in `spec/layout.py` order, appended to `weights.bin`, with their offsets | `compile()`, first |
| `--npu-encode` | `npu.mlir` | one ISA program per delegate id: `programs.bin` (words) + `programs.json` (tile plans, segments, relocations) | `compile()`, second |

The outlining between claim and pack (one module per delegate,
`func.call @__delegate__` in `cpu.mlir`) is not here: it is the back end's
generic `partition.cpp`.

**`driver/`: run time, C++, built on the board.** The runtime needs one
thing from the driver: that `__delegate__` reaches the NPU. Everything that
serves it is one C++ library, `libnpu_rt.so`; only the bitstream manifest
(Python, used by CD and before loading) and the legacy matrix runtime stay
beside it.

```
driver/
  npu_rt/                libnpu_rt.so, C++ inside, a C interface outside
    CMakeLists.txt
    npu_rt.h             extern "C" {
                           int npu_rt_load(const char *package_dir, uintptr_t mmio_base);
                           int npu_rt_execute(int64_t delegate_id, void **buffers, int count);
                           void npu_rt_close(void);
                         }
    load.cpp             once per package: read programs.json and weights.bin, allocate the
                         buffers the NPU reaches (CMA through PYNQ's libcma on the board, host
                         memory in sim), relocate each program's weight addresses, check the
                         overlay's array size against the package
    execute.cpp          per call: pack the INT8 input into A tiles, run the ISA program,
                         unpack the INT32 partial planes into the output buffer
    pynq.cpp             the board backend: map the NPU's registers through /dev/mem, write
                         PROG_ADDR / PROG_LEN / DATA_BASE, START, poll the status register
    sim.cpp              the export machine's backend: a C++ port of the ISA golden model
                         (src/test/model/isa_sim.py) over host memory
  overlay/               the FPGA bitstream (Python, unchanged)
    build.py             after Vivado: write the BIT/HWH provenance manifest
    verify.py            before loading (board) and before a release (CD): check BIT/HWH
                         against that manifest
  legacy/
    dma.py               the ABI 1.0 DMA matrix runtime, used only by the matrix demo and
                         src/deploy/matrix.py                                 (from driver/dma.py)
```

**Built on the board.** `libnpu_rt` is compiled on the PYNQ-Z1 itself, with
the g++ and CMake its image ships, against the board's own glibc, libstdc++
and `libcma`. No cross-compiler or sysroot is maintained, and the library
always matches the system it runs on. `src/deploy/` bundles the sources; the
board-side deployment step runs `cmake -S driver/npu_rt -B build && cmake
--build build` once per deployment, before the first package loads. On the
export machine the same sources build natively with the SIM backend, for
running packages on the PC; nothing is cross-compiled.

At run time on the board:

```
Python (src/runtime)   pynq.Overlay(...)                    load the bitstream (PYNQ's standard way)
                       npu_rt_load(package, mmio_base)      via ctypes, once per package
                       register delegate device "npu" -> npu_rt_execute
model.so: decode()  -> func.call @__delegate__(3, ...)      the runtime's small C dispatcher
                    -> delegate table: 3 is "npu"           -> npu_rt_execute(3, buffers)
                    -> pack, START, poll, unpack            -> result back to model.so
```

Python keeps loading the bitstream (`pynq.Overlay`) and calling the C
interface through ctypes; pybind11 is not used, so the board needs no Python
headers for the build. The runtime (`src/runtime/`) owns `__delegate__`, the
dispatcher and the registry; the driver never imports or links
`src/runtime/`.

| Change | Why |
|---|---|
| `isa/` -> `spec/`, plus `quant.py` and `board.py` | Every hardware fact in one place: the front end reads the quantization formats, the back end the board's CPU. `target.py` sat in `compiler/` although Cortex-A9 is a fact about the board |
| `isa/sim.py` -> `src/test/model/isa_sim.py` | Only verification uses it (`test_isa.py`, `generate_isa_vectors.py`); it is the golden model the RTL must match bit for bit, which is what `src/test/model/` holds. Running packages on the export machine uses `npu_rt`'s C port, not this file, so the board package no longer ships it |
| `compiler/` -> `custom_dialect/` | C++ only: the dialect and the passes the npu delegate adapter calls |
| `Partition.cpp` -> `Claim.cpp` | Claiming only; outlining is the back end's generic pass |
| `Lower.cpp`: `EncodePass` -> `Encode.cpp`; `HostLowerPass` deleted | `__delegate__` calls replace `--npu-lower-host` |
| `npu.call`, `npu.module` deleted from the dialect | Same reason; `npu.mlir` is a plain module |
| `package.py` -> `custom_dialect/lib/Pack.cpp` | torch-mlir puts weights in the IR as constants, so packing reads them in a pass; no Python weight dict exists any more |
| `passes.py` deleted | Building and running the tool moves to `pipeline.py`; `lower_host()` and `RUNTIME_C` go with the pass and with linking `npu_rt` into `model.so` |
| `npu_rt.c`, `placement.py`, `memory.py`, `abi.py` -> `driver/npu_rt/` (C++) | One library does all the delegate needs: load (buffers, relocation, geometry check) and execute; no ctypes struct mirrors left to drift |
| `driver/isa.py` deleted | `IsaDevice.run`, `gemm` and `NpuGemm` duplicate `pynq.cpp`; its capability and geometry check moves into `load.cpp`. `src/runtime/cli.py`'s reference GEMM path and its test switch to `libnpu_rt` |
| built on the board | Matches the board's glibc / libstdc++ / libcma without a sysroot; today's freestanding `npu_rt.c` avoided the problem only by using no libraries at all |
| `driver/dma.py` -> `driver/legacy/dma.py` | Only the matrix demo uses the ABI 1.0 runtime; marking it legacy keeps it out of the delegate path |

Boundary rules (`test_vendor_boundary.py`): `spec/` imports nothing;
`custom_dialect/` and `driver/` follow `spec/` only (both C++ ports of
`spec/layout.py` and `spec/isa.py`, each checked by a test against the Python
spec); nothing in `src/vendors/` imports, includes or links `src/export/` or
`src/runtime/`; the board package ships `spec/` and the `driver/` sources,
built there.

**Two simulators, one answer.** `src/test/model/isa_sim.py` (Python) defines
the correct result of every ISA program; `libnpu_rt`'s SIM backend
(`driver/npu_rt/sim.cpp`) runs
packages on the export machine. A test runs the same programs, including
`generate_isa_vectors.py`'s, through both and requires identical memory
afterwards, so the export machine never drifts from the golden model the RTL
is checked against. Moving the file into `src/test/model/` changes no
numbers; it is listed here because that directory changes only
deliberately (AGENTS.md).

`driver/board_smoke.py` has no reference in code, CD or docs; whether to
delete it is a separate decision outside this plan.

### 5.4 `pipeline.py` and `cli.py`

- `pipeline.py` says **what** happens: load, export_graph, to_mlir and backend in
  order, each timed into the manifest's `compile_seconds`. It takes Python
  objects (the parsed config) and returns the package path. Tests, notebooks
  and `src/deploy/` call it directly. For the back end it builds `pynqo-backend`
  when stale and runs it, passing the devices, the CPU models (the board's
  facts from `spec/board.py`) and the output directory.
- `cli.py` says **how it is invoked**: it parses arguments, reads the config
  file, applies flag overrides, builds the Python objects, calls
  `pipeline.export`, prints a JSON summary and returns an exit code.
  Per-stage subcommands read and write stage files and call the entry
  functions directly. Only `cli.py` knows about argparse.

### 5.5 YAML export configs

One file per model or variant in `src/export/configs/`, with one section per
stage. Each stage reads only its own section; the `quantize` block inside
`load` is ignored by targets without a spec.

```yaml
# src/export/configs/smollm.yml
load:
  source: HuggingFaceTB/SmolLM2-135M-Instruct
  framework: auto            # huggingface, detected from the source
  quantize:
    scheme: w8a8             # W8A8Config; or w4a8, as the target's spec allows
                             # AWQ: an offline checkpoint as source (decisions 3.3, 3.4)

export_graph:
  chunk: 16                  # prefill length
  max_seq: 512
  kv: f32                    # or int8

to_mlir: {}                  # torch-mlir defaults

backend:
  devices: [cpu, npu]        # [cpu] alone: the FP32 reference (decision 3.7)
  cpus: [native, arm-cortex-a9]   # native: this machine's model, e.g. x86_64-alderlake
```

```yaml
# src/export/configs/resnet18.yml
load:
  framework: torchvision
  arch: resnet18
  weights: IMAGENET1K_V1
  sha256: f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
  quantize:
    scheme: w8a8

backend:
  devices: [cpu, npu]
  cpus: [native, arm-cortex-a9]
```

```bash
python -m src.export.cli export smollm --out build/pkg/smollm               # configs/smollm.yml
python -m src.export.cli export --config src/export/configs/smollm.yml --out build/pkg/smollm
python -m src.export.cli export smollm --scheme w4a8 --group 64 --kv int8 --out build/pkg/smollm-w4
```

A variant used once is a flag override; a variant used repeatedly becomes its
own file.

Rules:

1. **Precedence:** built-in defaults < config file < command-line flags.
2. **Strict validation:** each section maps onto one dataclass (the export
   shape, the quantization scheme, ...); an unknown key or a wrong type is an
   error, never ignored. Standard dataclasses suffice; no schema library.
3. **Entry functions never read YAML.** Only `cli.py` and `pipeline.py` do;
   every entry function stays callable from Python with plain arguments.
4. **Reproducibility:** the effective config (file plus overrides) is written
   into the package's `manifest.json`.
5. **Names are shortcuts:** `export smollm` resolves to
   `configs/smollm.yml`, so the existing command names in CD, examples and
   READMEs keep working.
6. **Dependency:** PyYAML, listed in the export requirements (the tests
   already import `yaml`).

`src/export/configs/` is a new tracked directory; `docs/rules/filetree.md` is
updated with it.

### 5.6 `src/deploy/`: board deployment bundles

`src/export/delivery/` is not part of the export flow: `matrix.py` bundles the
matrix demo, which exports no model at all, and `resnet18.py` runs after
export, bundling a compiled package with the runtime sources, the overlay and
the example notebook into a hashed zip. It moves to `src/deploy/`, beside
`src/export/` (host-side compilation) and `src/runtime/` (board-side
execution):

```
src/deploy/
  resnet18.py            compiled package + runtime + overlay + notebook -> zip; --check-only
  resnet18_assets.py     the demo sample image and gallery download (from model_arch/resnet18/download.py)
  resnet18-*.json        pins for those images
  matrix.py              the matrix demo's deployment directory
```

It is kept, not deleted: the release flow depends on it. CD runs
`python -m src.deploy.resnet18` (and `--check-only`) and
`python -m src.deploy.matrix` to build the bundles a draft Release carries;
`examples/resnet18/README.md` and `examples/matrix-multiplication/README.md`
tell users to run them; `test_resnet18_delivery.py` and
`test_matrix_delivery.py` cover them. All of these follow the new module
paths.

`resnet18.py` must follow the new package as well: `validate_model` checks
`manifest["libraries"]`, which become per CPU model with a role field
(section 5.3c); the manifest gains the delegate table; and the bundle must
carry the `driver/npu_rt/` sources, which the board-side deployment step
builds into `libnpu_rt.so` (section 5.3d), since `model.so` no longer
contains the NPU runtime. `matrix.py` copies vendor files by path, so its list follows the
vendor's new layout (`driver/legacy/dma.py`, `driver/overlay/verify.py`).
Adding a top-level directory under `src/` is a tree change, so
`docs/rules/filetree.md` and the directory table in `AGENTS.md` are updated
with it.

## 6. Phases

Every phase ends with its acceptance check; Phase 0 decides whether the plan
continues as written.

### Phase 0 — feasibility (stop and re-plan on failure)

Answered; see "Implementation status" for the results. Recorded as asked:

- Pin torch, transformers, torchao, iree-turbine and the IREE compiler
  together (iree-turbine was dropped; see answer 5).
- Tiny SmolLM end to end: transformers load quantized by a minimal
  `W8A8Config`, torch.export (StaticCache decode), `run_decompositions`,
  IREE to torch dialect and then linalg.
- Answer:
  1. Do torchao's quantized ops (`_int_mm` and the scale multiplications)
     survive export and the default decompositions, and does IREE
     lower them to an INT8 contraction `--npu-claim` recognises?
     If not, decision 3.5 falls back to A.
  1b. Does the pinned transformers accept our `AOBaseConfig` subclass in
     `TorchAoConfig`?
  2. Does `torch.utils.cpp_extension` build `aten/` on the export hosts
     (WSL/Linux gcc, macOS clang), and does the stable ABI work with the
     pinned torch?
  3. Does `pynqo-backend` (built against LLVM 22) parse IREE's output (built
     against IREE's own LLVM)? If not: hand over in MLIR's generic form,
     build `pynqo-backend` against IREE's LLVM, or make the back end an IREE
     compiler plugin. 3b. In what form does IREE emit the weights (dense
     constants or `dense_resource`), so `partition.cpp` can read
     them?
  4. Can the KV-cache update stay in place?
  5. Can the IREE compiler stop after input conversion and emit linalg? If
     not, use torch-mlir's linalg pipeline for that step (section 5.3b).
  6. Are iree-turbine and IREE compiler wheels available for Python 3.12 on
     WSL/Linux and macOS?
- Fallback if IREE blocks: torch-mlir directly (an LLVM incubator project;
  its snapshots target Python 3.10 and 3.11 and a PyTorch nightly). Taken in
  in rev.26: torch-mlir's own dated wheels, which now include Python 3.12.

### Phase 1 — quantization: spec, `frontend/quantization/torchao/`, `frontend/load/check.py`

- `src/vendors/npu_pynq/spec/quant.py` (W8A8, W4A8, as data);
  `frontend/quantization/torchao/config.py` and `custom_quant/w8a8.py`, `w4a8.py`;
  `frontend/load/check.py`; the `quantization` and `require` parameters of `load` in
  both adapters.
- Acceptance:
  - weights and activations quantized by our configs equal `src/test/model`
    and the current `quantize_weight` bit for bit, for W8A8 and W4A8
    g8/g64/g128;
  - `load` with `require` rejects a float layer, an asymmetric INT8 layer and
    a layer quantized by another tool, each with a test, and accepts our
    configs' output;
  - `test_vendor_boundary.py` still passes (the spec imports neither torch
    nor `src/export`).

### Phase 2 — the C++ back end and the vendor

- The `__delegate__` ABI (buffers, outputs, errors) and the manifest's
  delegate table, written down first.
- `src/vendors/npu_pynq/` reorganised (section 5.3d): `spec/`,
  `custom_dialect/` (`--npu-claim`, `--npu-pack`, `--npu-encode`; `npu.call`,
  `npu.module` and `--npu-lower-host` deleted); `driver/npu_rt/` in C++
  (load, execute, the PYNQ and SIM backends), built on the board and
  natively on the export machine; `driver/isa.py`, `placement.py`,
  `memory.py`, `abi.py` deleted; `isa/sim.py` moved to
  `src/test/model/isa_sim.py`, with the test that it and `sim.cpp` agree.
- `backend/` C++ skeleton and `pynqo-backend`: the generic outline and
  weight-externalization passes, the npu delegate adapter, CPU codegen
  (in-process lowering to LLVM IR, then `llc` and the linker), the manifest;
  `lit` + `FileCheck` tests per pass. In-place KV cache (and, only with
  decision 3.5 A, the custom-op lowering in `frontend/to_mlir/custom.py`).
- Runtime: `__delegate__` (a small C dispatcher), the backend registry,
  loading `libnpu_rt` through its C interface after `pynq.Overlay`, reading
  the delegate table; `src/runtime/cli.py`'s reference GEMM through
  `libnpu_rt`.
- Board: the deployment step builds `libnpu_rt` with the board's g++ and
  CMake; the time it takes is recorded.
- Acceptance: tiny SmolLM gives the same NPU task count as today, now as
  delegate calls; `model.so` links no NPU code; on a machine with LLVM 22,
  host-simulated logits match the current package (integer paths bit for
  bit, float paths within tolerance); a test delegate registered only in
  tests proves a second device needs no change to the generic passes, CPU
  codegen or `model.so`; `test_vendor_boundary.py` passes with the new
  layout.

### Phase 3 — SmolLM2

- `configs/smollm.yml`; the FP32 reference is the same model with
  `devices: [cpu]` and no quantization.
- Acceptance:
  - FP32 build equals transformers (24/24 greedy tokens, as recorded in
    docs/goals/sw_optimal/Results.md);
  - W8A8, W4A8 and INT8-KV builds generate what the current packages generate
    on the host;
  - the CD frozen-output check (`.github/cd/smollm.py`) passes;
  - tokens per second on the host compared with the current package and
    recorded.

### Phase 4 — ResNet-18

- TorchVision model loaded by name from `configs/resnet18.yml`, BatchNorm
  folded by `to_mlir`'s layer fusion (section 5.3b), convolutions through
  `convert_conv2d_to_img2col` onto the NPU.
- The demo sample image and gallery downloads move to `src/deploy/`; CD, the
  ResNet example README and notebook, and `test_resnet18_sources.py` follow
  (the checkpoint download step disappears: export downloads the weights).
- `src/deploy/resnet18.py` follows the new package (section 5.6) and bundles
  the `npu_rt` library; `--check-only` still verifies an existing bundle.
- Weights are cached where the examples and CD expect them
  (`dist/resnet18/model`, through `TORCH_HOME` or a local `source`).
- Acceptance: top-1 unchanged on the five gallery images; NPU task count
  compared with today (21).

### Phase 5 — remove the old front end

- Delete the directories in section 5 and the old Python back end
  (`compile.py`, `targets.py`, `compilers/`); update the CLI, CD, examples
  (WSL and macOS guides together), `docs/rules/filetree.md` (including the
  vendor layout and its rules, and `src/deploy/`), the directory table in
  `AGENTS.md`, and the layering tests.
- Replace `src/test/vectors/export_golden.json` (it pins the old MLIR) with:
  quantization bit-exactness against `src/test/model` (Phase 1) and per-model
  logits checks (Phases 3-4).

## 7. Risks

| Risk | Mitigation |
|---|---|
| torch-mlir: its LLVM differs from `pynqo-backend`'s LLVM 22; a lowering that leaves a non-upstream op | Resolved in Phase 0: the output is upstream-only and LLVM 22 parses it; models avoid ops that lower outside upstream (`index_put`) |
| The back end in C++ iterates more slowly than Python and needs MLIR C++ experience | Small passes with `lit` tests; opt-style single-pass runs for debugging; CPU codegen keeps `llc` in the first version |
| torchao's quantized tensor cannot express the numeric contract exactly | Phase 1 bit-exact checks; then an own tensor subclass in `custom_quant/`, or decision 3.1 revisited before Phase 2 |
| torchao's API moves (it is winding down new features) | Pinned version; torchao-specific code confined to `frontend/quantization/torchao/` and `frontend/load/check.py` |
| Replaced kernels change results process-wide | `cpu/` kernels are tested against PyTorch's own; loading them is explicit (`load_kernels()`), never implied by export |
| The export host needs a C++ compiler, and the library is tied to the torch ABI | Built on demand by `torch.utils.cpp_extension` on WSL/Linux and macOS; stable ABI checked in Phase 0 |
| Export dependencies grow (torch, transformers, torchao, torch-mlir) | Pinned in one requirements file for the export host; the board runtime is unaffected |
| Performance regressions: online attention, INT8 KV and other tuned kernels become generic lowering | Tokens per second compared per model in Phases 3-4 before the old path is removed |
| Qwen3-MoE routing exported as "run every expert" | Deferred with Qwen3; revisit when Qwen returns |
| Building `libnpu_rt` on the board: the image's g++ / CMake versions, build time, root for `/dev/mem`, `libcma` availability | Phase 2 builds and runs it on the board and records versions and time; the deployment step fails clearly if a tool is missing |
| The `__delegate__` ABI is hard to change once packages ship | Defined and reviewed first in Phase 2; the manifest records its version |
| No safety net while the golden is replaced | Phases 1-3 checks land before Phase 5 removes anything |
| This machine has neither torch nor LLVM 22 | Phase 0 onwards runs on WSL/Linux or CI |
