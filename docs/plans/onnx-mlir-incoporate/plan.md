# Incorporate ONNX-MLIR into the SmolLM2 Export Pipeline

Status: implementation started; source inventory and import are implemented.
ARM code generation and the unchanged-NPU compatibility gate remain incomplete.

## Objective

Compile the pinned SmolLM2 INT8 ONNX model into a package containing ARM CPU
code, NPU programs, weights, and tokenizer assets through one export command.
Replace the production ONNX Runtime operator sessions and their C bridge with
compiled CPU implementations produced through ONNX-MLIR.

Keep the existing NPU compiler interface, instruction set, driver, and RTL.
No changes under `src/vendors/npu_pynq/` or `src/hw/` are planned.

The requested plan directory is used explicitly instead of the usual dated
plan filename convention. This document records design and implementation
progress; it does not claim the production migration is complete.

## Planned Files and Directories

```text
src/export/
|-- cli.py                                  modify
|-- core/
|   `-- onnx_source.py                      modify
|-- graphs/smollm/
|   |-- model.py                            modify
|   `-- onnx.py                             modify
`-- compilers/onnx_mlir/                     add
    |-- __init__.py
    |-- toolchain.py
    |-- pipeline.py
    |-- support.py
    |-- toolchain.json
    `-- mlir/                               add if custom passes are needed
        |-- CMakeLists.txt
        |-- include/
        `-- lib/

src/runtime/core/
`-- package.py                              review; modify only if necessary

src/test/tests/
|-- test_onnx_import.py                      add
|-- test_onnx_lowering.py                    add
`-- test_onnx_package.py                     add

examples/smollm_onnx/
|-- README.md                               modify
|-- README.win.md                           modify
`-- README.macos.md                         modify

docs/rules/
`-- filetree.md                             modify
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
currently calls `attach_float_runtime`, and `core/onnx_source.py` implements
that function and bundles the ORT library, operator models, and initialization
hooks. Both must change to remove the production dependency.

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
  Existing NPU lowering        ONNX-MLIR CPU lowering
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

Update Linux, native Windows, and macOS guides consistently. State supported
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

OpenSpec change: `incorporate-onnx-mlir`, tasks 1.1-1.3 complete (3/11 total).

Implemented source inventory and real ONNX-to-MLIR import under
`src/export/compilers/onnx_mlir/`. A standalone module command exposes this
boundary before production export integration. The pinned SmolLM2 model and
a dynamic Sigmoid fixture both import successfully. Generated artifacts remain
ignored under `build/onnx-mlir/`.

The consumer-facing production exporter is unchanged. The imported model still
contains 151 custom operations requiring conversions. The pinned compiler image
uses LLVM 24 and rejects ARM object generation because it lacks an ARM target;
the existing NPU toolchain uses LLVM 22. Task 1.4 therefore remains blocked.
An ARM-enabled compiler build and a verified IR adapter are required before
continuing the production integration path. No vendor, ISA, RTL, or ORT source
files were modified or removed by this implementation stage.

Validation: 18 focused and boundary tests passed, including real Docker import.
Existing compiler/boundary regressions ran 19 tests with 8 tool-dependent skips.
Strict OpenSpec validation passed. CPU execution parity, NPU partition parity,
and PYNQ-Z1 execution remain unverified.

See `docs/onnx-mlir/README.md` for commands, artifact meanings, pinned revisions,
and the exact observed ARM compilation error. Before implementing the remaining
production units, follow task 2.1's issue/sub-issue ownership gate.
