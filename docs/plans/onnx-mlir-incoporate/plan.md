# Incorporate ONNX-MLIR into the SmolLM2 Export Pipeline

Status: implemented, validated and deployed on physical PYNQ-Z1. Production
ORT sources and dependencies have been removed after acceptance.

## Objective

Compile the pinned SmolLM2 INT8 ONNX model into a package containing ARM CPU
code, NPU programs, weights, and tokenizer assets through one export command.
Replace the production ONNX Runtime operator sessions and their C bridge with
compiled CPU implementations produced through ONNX-MLIR.

Keep the existing NPU compiler interface, instruction set, driver, and RTL.
No changes under `src/vendors/npu_pynq/` or `src/hw/` are planned.

The requested plan directory is used explicitly instead of the usual dated
plan filename convention. This document records design and implementation
progress. Evidence and reproduction are recorded in [acceptance.md](acceptance.md).

## Planned Files and Directories

```text
src/export/
|-- cli.py                                  modify: import/export orchestration
|-- core/onnx_source.py                      modify: remove ORT attachment
|-- graphs/smollm/{onnx,model}.py             modify: consume imported semantics
|-- compilers/mlir/builder.py                modify: preserve literal precision
|-- compilers/mlir/ops/attention.py           modify: optional fixed arithmetic
`-- compilers/onnx_mlir/
    |-- toolchain.{py,json}                 pinned invocation and provenance
    |-- pipeline.py, support.py             import and inventory
    |-- adapter.py                          source/IR identity and parsed graph
    |-- lowering.py, math_kernels.py         verified CPU conversions
    `-- mlir/
        |-- CMakeLists.txt
        `-- lib/Adapter.cpp                 MLIR 22 parser; no NPU passes

src/runtime/
|-- core/package.py                         remove ORT-specific fallback
|-- cli.py                                  reference consumer migration
`-- cpu/ort.py                              retain optional Python host reference

src/test/
|-- onnx_mlir_acceptance.py                  independent host/physical acceptance
`-- tests/{test_onnx_import,test_onnx_lowering,test_onnx_adapter,test_compiler}.py

examples/smollm_onnx/
|-- README.md                               WSL guide
|-- README.macos.md                         macOS deployment guide
`-- smollm_onnx.ipynb                        output-free board demo

.github/workflows/cd.yml                     compile without ARM ORT build
.github/cd/smollm.py                         independently reviewed fixed reference
docs/rules/filetree.md                      compiler/runtime responsibilities
```

Remove the following files only after their production consumers have been
migrated and validation passes:

```text
src/runtime/cpu/onnx_ops.c
src/runtime/cpu/onnx_ort_bridge.c
src/runtime/build_onnxruntime_arm.sh
src/runtime/build_onnxruntime_bridge.sh
```

### Generated Artifacts and Production Logic to Remove

Regenerate the production package without these artifacts:

```text
dist/onnx_compiled_ops.o
dist/smollm-package/libonnxruntime.so.1.20.1
dist/smollm-package/float-ops/*.onnx
```

The preparation workflow must no longer require the model workspace's
`onnx-runtime/libonnxruntime.so.1.20.1`, `onnx-runtime/onnxruntime_c_api.h`,
or `onnx-ops/*.onnx`. Existing workspace copies are optional cleanup after
checking reference and other consumers; do not delete user artifacts as part
of the source migration automatically.

Remove or replace the following production logic:

- ORT session creation, operator execution, error handling, and initialization
  through `onnx_compiled_init`, `onnx_compiled_clear`,
  `onnx_compiled_status`, and `onnx_ort_error`.
- The SmolLM call to `attach_float_runtime` and its ORT asset packaging,
  hashes, preload configuration, and initialization hooks.
- The SmolLM requirement for `onnx_compiled_ops.o` through `--native-object`.
  Retain generic native-object support if other compilation paths use it.
- Production preparation steps that generate ORT operator-session models,
  and documentation or packaging commands that build or copy ORT artifacts.

Delete source files only after replacement numerical tests and PYNQ-Z1 board
tests pass and the consumer audit finds no remaining dependencies. Preserve
the independent ONNX/ORT reference used for verification. If a proposed deletion
still serves that reference or another consumer, migrate the consumer first or
revise the deletion list explicitly. Keep generic package runtime hooks that
other packages still require.

The earlier file list omitted two integration points: `graphs/smollm/model.py`
previously called `attach_float_runtime`, and `core/onnx_source.py` implemented
that function and bundled ORT assets and hooks. Both were migrated.

Exact C++ filenames will follow the passes proven necessary by the compatibility
investigation. Do not create empty placeholder directories or duplicate upstream
implementations. Existing tests or example packaging scripts may need focused
updates if the consumer audit finds references to the removed files; record
those paths before implementation.

## Architecture and Boundaries

```text
Pinned ONNX model and source verification
                    |
       ONNX-MLIR import: ONNX dialect MLIR
                    |
       Operator decomposition and IR adaptation
                    |
       Compile-time CPU/NPU partition
          |                         |
  Selected NPU computations    Remaining CPU computations
          |                         |
  Existing NPU lowering        Export-side CPU decomposition
          |                         |
  NPU programs and weights     LLVM and ARM code generation
          |                         |
          +---- package and link ---+
                    |
       Existing runtime package entry points
                    |
       Runtime dispatch by compiled ARM code
          |                         |
  Submit NPU programs          Execute CPU computations
```

This diagram is the intended boundary, not a claim that upstream ONNX-MLIR
already emits the input required by this repository's NPU passes.

ONNX-MLIR first imports the model into ONNX dialect MLIR. This can be an
in-memory module or an intermediate `.mlir` file; a separate file is useful
for inspection but is not required for every stage. Operator decomposition
and export-side adaptation must establish the input contract expected by the
existing NPU partition pass before invoking it.

Partition is a compile-time decision: identify supported contractions, assign
them to the NPU, and leave other computations for CPU lowering. Run it before
lowering destroys the high-level tensor operations needed for selection.
ONNX-MLIR does not automatically provide a backend for this repository's NPU.

Dispatch happens at runtime: the compiled ARM control flow executes CPU code
and submits the generated NPU programs through the existing driver, observing
buffer ownership and completion dependencies before consuming results. It does
not import or repartition the ONNX graph on the board. Synchronization must
preserve the current driver contract; asynchronous overlap is outside this
migration's scope.

- `compilers/onnx_mlir/` owns tool invocation, generic operator conversion,
  compatibility diagnostics, and pipeline orchestration. It must not import
  model frontends or vendor modules. Vendor callbacks are supplied by the
  export orchestration through the existing target registry.
- `graphs/smollm/` owns model configuration, named entry points, prefill lengths,
  decode behavior, and the FP32 KV-cache contract.
- Existing vendor code continues to own NPU selection, tiled weights,
  instructions, and hardware access. New adapters belong on the export side.
- `runtime/` loads compiled artifacts and runs generation. It must not import
  export code or require compiler binaries on the board.

CPU code may still call compiled helper functions, math libraries, and minimal
ONNX-MLIR runtime support. The requirement is to eliminate ORT session execution
and its bridge, not to eliminate every function call or all runtime support.

## Implementation Steps

### 1. Establish the Baseline and Compatibility Gate

1. Record the current source hash, package entry-point signatures, partition
   report, NPU programs, weight layouts, numerical results, and board timings.
2. Enumerate actual nodes by domain, operation, opset, dtype, shape, and attributes
   from the pinned ONNX file. Include Microsoft-domain fused operators.
3. Audit every consumer of ORT bridge files, native-object arguments, operator
   assets, and runtime hooks before scheduling deletion.
4. Select and pin an ONNX-MLIR revision and its required LLVM revision in
   `toolchain.json`. Check compatibility with the existing MLIR toolchain and
   `npu-opt`; do not assume different MLIR versions can exchange arbitrary IR.
5. Prove a small CPU graph can be compiled for Cortex-A9 ARM32 hard-float with
   the correct sysroot and linked into a library loadable on the board.
6. Prove an exported quantized contraction can be represented in the exact IR
   form accepted by the unchanged NPU partition pass. Keep weight metadata,
   signedness, zero-point handling, and accumulator semantics intact.

If a compatible IR boundary or ARM32 build cannot be demonstrated, stop that
integration path and document the blocker. Revise the plan before expanding
scope into vendor compiler changes. ONNX-MLIR integration is not yet a verified
drop-in replacement.

### 2. Add Toolchain Discovery and Operator Diagnostics

- Implement `toolchain.py` to locate pinned tools, check their versions, invoke
  them with structured arguments, and report actionable failures.
- Implement `support.py` to check the complete source graph, including custom
  domains, quantization types, dynamic dimensions, and cache semantics.
- Report unsupported operators before expensive compilation. Never silently
  fall back to ORT in the final production path.
- Keep downloaded source, tool builds, intermediate IR, and binaries in ignored
  build workspaces. Track version pins and authored integration code only.

### 3. Replace CPU Operator Sessions with Compiled Computation

Start with isolated operator cases, then integrate them into model entry points.
Use upstream lowering where it matches the required semantics; add generic
decompositions or custom passes only for demonstrated gaps.

| Operator family | Required behavior to preserve |
| --- | --- |
| RMS normalization and fused residual normalization | Epsilon, reduction axes, accumulation type, residual outputs |
| Rotary embedding | Position offsets, cache indexing, head layout, rotation convention |
| Group-query attention | Query/KV head mapping, causal mask, scaling, cache updates, prefill and decode |
| Dynamic quantization | Range handling, rounding, saturation, zero points, output types |
| Sigmoid and floating-point MatMul | Numerical tolerance, shapes, and output layout |

Replace the `L.native(..., 'onnx_*', ...)` ORT-dependent calls with compiled IR
or compiled functions produced by the new pipeline. A generated function call
is acceptable; forwarding the work to an ORT session is not.

Preserve the pinned INT8 weights and existing numeric contract. Do not apply
re-quantization or alter FP32 KV storage as part of this migration.

### 4. Integrate Export while Reusing the NPU Backend

1. Implement `pipeline.py` around the IR boundary proven in step 1: ONNX import
   to MLIR, required decomposition and adaptation, existing CPU/NPU partition,
   then target-specific lowering. Preserve inspectable intermediate IR for
   each boundary. Perform NPU selection before CPU lowering erases the relevant
   tensor contractions.
2. Preserve the existing NPU partition/lower inputs and output formats. Verify
   that intended NPU work has not accidentally become CPU work.
3. Update `graphs/smollm/onnx.py` and `model.py` to use compiled CPU operations
   and stop attaching ORT assets and hooks.
4. Update `cli.py` to orchestrate compilation, weight packing, linking, and
   packaging without requiring a user-supplied `--native-object` for this path.
   Preserve unrelated models and existing generic CLI behavior.
5. Update `core/onnx_source.py` so preparation can fetch and verify the source
   without requiring ORT operator-session generation. Retain graph/tensor export
   when it serves independent verification or other consumers.

The intended user command remains:

```bash
python -m src.export.cli export smollm \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --out dist/smollm-package \
  --prefill-lengths 36,40,42
```

### 5. Preserve Package Loading and Remove Production ORT Dependencies

Maintain existing runtime entry-point names and ABI where possible. Review
`runtime/core/package.py`; change it only if required for compiled package
metadata or compatibility. Keep generic runtime hooks if other packages need
them rather than removing them solely because SmolLM no longer does.

The final package should contain the ARM model library, weights, NPU programs,
tokenizer assets, manifest, and explicitly required minimal runtime libraries.
It must not require `libonnxruntime.so`, `float-ops/*.onnx`, or
`onnx_compiled_init` and related ORT hooks.

Keep an independent ONNX/ORT reference available for tests when useful. Do not
delete shared bridge files while reference or other production consumers still
depend on them; first migrate those consumers or revise the removal list.

### 6. Update Documentation and Complete Validation

Update the WSL and macOS guides consistently with the current create-example rule. State supported
build hosts and how other platforms obtain matching artifacts; do not imply
native ARM cross-compilation is supported before it is tested. Preserve the
notebook as the human validation entry point. Update `docs/rules/filetree.md`
for the new compiler directory and revised CPU runtime responsibilities.

## Validation and Acceptance

- Import tests verify source identity, full operator inventory, unsupported-op
  diagnostics, and prefill/decode shapes.
- Lowering tests compare isolated operators with the independent reference,
  including zero/constant quantization inputs, multiple positions, GQA head
  mapping, and cache boundaries. Quantized values must meet the existing exact
  contract; floating-point tolerances must be specified before comparison.
- Package tests verify ABI, entry points, assets, required libraries, and the
  absence of production ORT symbols and session files. Inspect ARM library
  headers, symbols, and dependencies, including ARM32 hard-float compatibility.
- Integration tests compare logits, KV state, and deterministic generation for
  prefill lengths 36, 40, and 42 followed by multiple decode steps. Compare
  numerical tensors as well as generated text.
- Compare baseline and new partition reports, NPU instruction streams, and
  weight layouts for unchanged NPU computations. Explain any differences.
- Run existing software compiler, package, vendor-boundary, and JIT-boundary
  regressions. Run `make -C src/test lint sim` before any eventual push, as
  required by the repository.
- On PYNQ-Z1, load the existing overlay and run the compiled package without
  installing ORT. Record correctness, peak memory, prefill latency, and decode
  throughput against the same baseline workload. Do not claim improved speed
  merely because the bridge was removed.

Completion requires a working single-command export, unchanged NPU interfaces,
independently validated CPU semantics, and real board execution without the
production ORT dependency. Until the board run is complete, report that part as
unverified.

## Implementation Progress: 2026-10-25

The pinned LLVM 24 importer emits generic ONNX IR, which the export-side MLIR 22
adapter parses. The existing LLVM 22 toolchain generates ARM32 code. The import
image's missing ARM backend does not require a vendor upgrade.

Production export consumes imported operations, attributes and SSA edges before
NPU partition. CPU normalization, quantization, rotary embedding, GQA, Sigmoid
and the floating head compile to ordinary MLIR/LLVM operations. NPU programs,
all 480 tasks, weights and layouts match the legacy path byte for byte.

Host and PYNQ-Z1 passed 12 numerical cases: each prefill profile 36/40/42 plus
three decode steps. KV state and greedy IDs match exactly; max logit error is
4.2e-5. The public notebook generated 32 independently verified tokens on the
physical board. ONNX Runtime is absent from the board's PYNQ Python environment.

The four production ORT files and 13 unused ORT-only experiment build/debug
wrappers were removed after board validation. Current export/runtime/CD paths
have no consumers of them. The host Python ORT reference, board NumPy/NPU
reference, source graph export and explicit `prepare-onnx operators` remain.
Generic native objects and runtime hooks remain available to other callers.
Existing downloaded artifacts and historical result records were preserved.

Sub-issues #139, #140 and #141 are assigned to yenhao-huang and claimed by
agent a, with dedicated worktrees created from dev and advanced to the required
#119 prerequisite. Integration stays in the existing #119 worktree.
No vendor, ISA, driver or RTL source changed. No push, PR, merge or release
was performed.

See [README.md](README.md) for commands and [acceptance.md](acceptance.md) for
numerical, physical, reference-review and removal evidence. OpenSpec tasks
remain the authoritative completion checklist.
