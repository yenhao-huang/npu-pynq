# Delegate framework plan

| Field | Value |
| --- | --- |
| Plan ID | 2026-1113-001 |
| Topic | Devices become opaque delegates: the middle end cuts each device subgraph out and puts a delegate call in its place; each device back end compiles its subgraphs on its own, and CPU code generation never sees inside one |
| Date | 2026-11-13 |
| Status | Draft |
| Based on | `npu/issue177-b` at `6a513db` (PR #178: assignment and partitioning already in the middle end) |
| Related | 2026-1112-005 (middle end / back end split), 2026-1112-004 (kernel outlining), 2026-1112-002 (kernel profiler) |
| Issues | to be opened |

## Problem

After #178 the middle end outlines each NPU contraction into
`@npu_<id>`, but the device stays visible to the rest of the compiler:

- `ir/model.mlir` holds the device bodies next to the CPU graph.
- The back end's `partition.cpp` asks the delegate to lower each body
  (`npu.matmul {tk}` plus a `linalg.reduce` over K planes), inlines the
  result into the CPU graph, and only then replaces `npu.matmul` with
  `__delegate__`.
- The K-plane `linalg.reduce` is NPU knowledge (the NPU splits K at `MAX_K`
  and returns `[Kt, M, N]` partial planes) that ends up as CPU code in
  `model.so`, and the delegate's output size in `compile.json`
  is `Kt·M·N·4` bytes, not the subgraph's result.

So the CPU code generator, the delegate table and the runtime ABI all depend on
how the NPU happens to tile K. A delegate framework (ExecuTorch
`to_backend` + `executorch_call_delegate`, LiteRT delegates) treats a device
as a black box instead: the framework knows which nodes a device takes and
what goes in and comes out, never how the device computes them.

## Decision

Adopt the delegate model for devices:

1. **Middle end — partition and delegate.** The device's partitioner (its
   match predicate, already `npu::isClaimable` in `NpuMatch`) picks the ops.
   The middle end moves each subgraph into a separate device module and
   leaves an opaque, bodiless delegate declaration in the CPU graph. The
   graph-level passes (fusion) then run on a CPU graph that has no device
   bodies in it.
2. **Back end — per-device code generation.** Each device compiles its own
   module into artefacts (data for the NPU). CPU code generation compiles
   the CPU graph and lowers each delegate declaration to the fixed runtime
   ABI `__delegate__(id, in, out)`. Nothing is inlined back.
3. **Runtime — the delegate owns its host work.** Whatever the device needs
   on the host (packing, unpacking, summing K planes) lives in the device's
   runtime library (`libnpu_rt`), built once, not in each `model.so`.

Why partitioning and the delegate call sit in the middle end, which is
otherwise target-agnostic: both are whole-graph decisions. Where the cuts go,
which neighbours share a device, one global delegate id per subgraph, and
every later graph pass (fusion) must respect the cuts; only the middle end
sees the whole graph and every device. A back end compiles one device's
subgraphs and never needs the whole graph. The middle end stays generic: the
device only plugs in its match predicate (as LLVM's `TargetTransformInfo`,
TVM's `target.<compiler>` op attributes, ExecuTorch's `Partitioner`). IREE
(`stream` affinity before `hal`), TVM BYOC (`PartitionGraph` before codegen)
and ExecuTorch (`to_backend` before `preprocess`) all partition ahead of
code generation for this reason.

This replaces the IREE-style placement of 005 phase 2 for device code
(IREE keeps device executables in the shared IR until `hal`); the stage split
for CPU code (graph work in the middle end, bufferization and below in the
back end) stays. `docs/rules/lib/export_stages.md` is updated accordingly.

## Code generation depth

| | Input | Stops at | Never emits |
| --- | --- | --- | --- |
| CPU (`--cpus` targets) | `ir/model.mlir`: linalg on tensors with delegate declarations | a shared library per target (`<cpu>/model.so`): bufferization → loops → LLVM dialect → LLVM IR → `llc` → link; each delegate call becomes `__delegate__(id, in, out)` on byte buffers | anything that depends on which device an id belongs to: no tile sizes, no K planes, no packing |
| NPU | `ir/delegates/npu.mlir`: one `@npu_<id>` per delegate, linalg on tensors | device binaries and data: `npu.matmul {tk}` → `GemmPlan` → ISA program in `programs.bin` (`.txt`, `.json`) and B tiles in `weights.bin`; one `delegates` row per id with input and output bytes | host code: the NPU contributes nothing to `model.so` |

Why the NPU stops there: it is the ExecuTorch / LiteRT shape (an AOT blob per
delegate, a runtime that executes it), and our runtime has no MLIR, so
anything less than an encoded program would need a JIT. Why the CPU stops at
`model.so`: the schedule stays compiled code (plan 004's known deviation); a
delegate is just an external call to it.

The delegate's contract is the subgraph's signature: `in` holds the
activation (`M×K` INT8), `out` the subgraph's result (`M×N` INT32), so
`output_bytes = M·N·4`. Tiling over K is internal to the NPU.

## Design

### 1. Delegate result: the NPU returns the full sum

- `npu.matmul` is redefined to return `M×N` INT32 (the sum over K), with
  `tk` still an attribute. `npu::claim` no longer creates the
  `linalg.reduce`.
- `libnpu_rt` (`execute.cpp` `unpackC`, and `sim.cpp`) accumulates the K
  planes into `out` while unpacking C instead of writing `[Kt, M, N]`. The
  sum uses wrapping 32-bit addition (unsigned arithmetic in C), the same as
  today's `arith.addi`, so outputs stay bit-identical.
  - Open: `src/test/model/numeric.py` has `saturating_add_int32`; the
    hardware's in-chunk accumulation and today's cross-chunk wrap differ in
    overflow behavior. This plan keeps today's behavior; the numeric
    contract question is recorded, not changed.
- The delegate's accumulator init must be zero: `isClaimable` additionally
  requires `outs` to be `linalg.fill` of 0 on `tensor.empty` (all 422 + 21
  claimed matmuls today are). A non-zero init would need a second input;
  such ops stay on the CPU.
- `compile.json` `delegates[].output_bytes` becomes `M·N·4`; `format` is
  bumped and the runtime rejects the old format.
- The profiler's `unpack_c` phase now includes the sum; no new field.

### 2. Middle end: delegate declarations and device modules

`pynqo-partition-devices` (from #178) changes what it leaves behind:

```mlir
// ir/model.mlir (CPU graph)
%8 = call @__delegate_0(%7) : (tensor<12544x147xi8>) -> tensor<12544x64xi32>
func.func private @__delegate_0(tensor<12544x147xi8>) -> tensor<12544x64xi32>
    attributes {pynqo.delegate_id = 0 : i64, pynqo.device = "npu"}

// ir/delegates/npu.mlir (device module)
func.func @npu_0(%arg0: tensor<12544x147xi8>) -> tensor<12544x64xi32>
    attributes {pynqo.delegate_id = 0 : i64} {
  %w = arith.constant dense_resource<...> : tensor<147x64xi8>
  %0 = tensor.empty() ...  %1 = linalg.fill ...  %2 = linalg.matmul ...
  return %2
}
```

- The declaration has no body, so canonicalize, CSE and fusion cannot look
  into or across it. `pynqo.device` stays only so the back end can route and
  check; CPU code generation ignores it.
- `pynqo-middle-end` gains `--delegates-dir` (or writes next to `-o`);
  `pipeline.py` passes the directory to the back end. One module per device,
  ids global and `0..n-1` as today.
- Weights used by a delegate live in its device module only. A constant both
  sides read is in both modules (today's arena already stores both forms).

### 3. Back end: device compile, then CPU codegen

- `Delegate` interface: `lower(func::FuncOp)` is removed; it becomes
  `compile(ModuleOp deviceModule, WeightArena &, out)` returning the
  `delegates` rows (id, index, input bytes, output bytes). The NPU delegate
  runs `npu::claim` on its own module and encodes as today.
- `partition.cpp` is deleted. A small CPU-codegen pass,
  `lower-delegate-calls` (under `backend/cpu/`), runs after
  `one-shot-bufferize` and `buffer-results-to-out-params`: each call to a
  `@__delegate_<id>` declaration becomes the byte-view sequence and
  `func.call @__delegate__(%id, %in, %out)`; the declarations are erased.
  It reads only `pynqo.delegate_id` and the memref types.
  - It may avoid today's copy of `%x` into a fresh buffer when the operand is
    already contiguous; measured, not promised.
- `pynqo-backend` checks: every declaration has a body in exactly one device
  module, ids cover `0..n-1`, devices match `--devices`.

### 4. Rule and plan updates

- `export_stages.md`: the "whole-graph decisions are middle end; per-device
  compilation is back end" rule is added with #178. This plan adds "Devices
  are delegates (ExecuTorch / LiteRT): a device back end compiles only its
  own module and contributes no CPU code", updates the stage table (the
  middle end inserts the delegate call; the back end no longer inlines), and
  drops the "until it lands" note. Keep "follow IREE" for where CPU-path
  steps go; record the device-code deviation.
- `filetree.md`: `backend/partition.cpp` removed, `backend/cpu/` gains the
  delegate-call lowering, package `ir/delegates/`.
- Plan 004: kernel outlining runs on the CPU graph only; a delegate
  declaration is already an outlined kernel (the profiler's `__delegate__`
  hook covers it).
- Plan 005: phase 2's "partition inlines the lowered body" is superseded.

## Package layout

```text
package/
├── ir/model.mlir        M  CPU graph, @__delegate_<id> declarations, no device bodies
├── ir/delegates/npu.mlir A  one @npu_<id> per delegate
├── compile.json         M  delegates[].output_bytes = M·N·4; format bumped
├── programs.*           -  same programs (encoding unchanged)
├── weights.bin          -  unchanged
└── <cpu>/model.so       M  no K-plane reduce; calls __delegate__ only
```

## Phases

| Phase | Scope | Acceptance |
| --- | --- | --- |
| 1 | Delegate result is the full sum: `npu.matmul` returns `M×N`, `libnpu_rt` sums K planes (wrapping), claim requires a zero init, `compile.json` format bump | SmolLM2 24 greedy steps and ResNet-18 outputs bit-identical to `dev` on the host sim path; same delegate count and ids; `programs.bin` byte-identical; `model.so` has no K-plane reduce; `src/test` runtime tests for multi-plane delegates (`K > MAX_K`); lint, sim, export tests pass; board run of both demos |
| 2 | Middle end writes declarations + `ir/delegates/<device>.mlir`; back end `Delegate::compile(ModuleOp)`, `lower-delegate-calls`, `partition.cpp` deleted | Same outputs, delegate table and `programs.bin` as phase 1; `ir/model.mlir` has no device ops or bodies; `pynqo-backend` CPU codegen links no `NpuDialect` code path for the CPU graph (layering test); mismatched modules fail with a clear message; `export_stages.md`, `filetree.md`, plans 004/005 updated |

Phase 1 is independent of #178; phase 2 builds on #178 (merge it first,
it already moved the partitioner into the middle end).

## Risks and open questions

- **Runtime ABI change.** Old packages (planes) and new runtime do not mix;
  the `format` bump makes it an error instead of wrong numbers.
- **Host time.** Summing in `unpackC` replaces a separate compiled
  `linalg.reduce` over the same data; expected neutral or faster (one pass,
  smaller `out`). Measured with the profiler on the board in phase 1.
- **Overflow semantics** (section 1): kept as today; whether the cross-chunk
  sum should saturate is a numeric-contract decision for `src/test/model/`,
  out of scope.
- **Two outputs per delegate later.** A device that needs several inputs
  (e.g. a non-zero accumulator or bias) needs a multi-buffer `__delegate__`;
  not needed for the NPU today.
- **Debuggability.** Device bodies leave `ir/model.mlir`; `--keep-ir` keeps
  `ir/delegates/` so both halves stay inspectable.

## Not in scope

- New devices, cost-based placement, more than one NPU.
- Changing the ISA encoding, `programs.bin` format or RTL.
- Kernel outlining of the CPU graph (plan 004).
