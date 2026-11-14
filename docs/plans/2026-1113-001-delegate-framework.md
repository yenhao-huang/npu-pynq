# Delegate framework plan

| Field | Value |
| --- | --- |
| Plan ID | 2026-1113-001 |
| Topic | Devices become opaque delegates: the middle end cuts each device subgraph out and puts a delegate call in its place; each device back end compiles its subgraphs on its own, and CPU code generation never sees inside one |
| Date | 2026-11-13 |
| Status | Done in #178 (both parts, one change); board run pending |
| Based on | `npu/issue177-b` at `6a513db` (PR #178: assignment and partitioning already in the middle end) |
| Related | 2026-1112-005 (middle end / back end split), 2026-1112-004 (kernel outlining), 2026-1112-002 (kernel profiler) |
| Issues | #177 (PR #178): partitioning and the delegate call land together |

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
- `compile.json` `delegates[].output_bytes` becomes `M·N·4`; its `format`
  becomes `pynqo-backend/2`, and the package format `pynqo-package/3`, which
  the runtime's loader checks, so a package built before the change is
  refused instead of returning partial planes as a result.
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
- `pynqo-middle-end` gains `--delegates-dir` (default: `delegates/` next to
  `-o`); `pipeline.py` passes `ir/delegates` to both tools. One module per
  device (`module @npu attributes {pynqo.device = "npu"}`), ids global and
  `0..n-1` as today.
- As built: `pynqo-partition-devices` puts each device's functions in a
  nested `module @<device>` and the declaration in the CPU graph, so the
  pass stays testable on its own. `pynqo-middle-end` then runs in two pass
  managers: up to partitioning, then it writes each nested module to
  `<delegates-dir>/<device>.mlir` and removes it, then canonicalize, CSE and
  fusion on the CPU graph alone. Device bodies therefore no longer go
  through those graph passes (they did with the earlier inline design); the
  NPU delegate reads only the matmul and its constant weight, so nothing it
  compiles changes.
- Weights used by a delegate live in its device module only. A constant both
  sides read is in both modules (today's arena already stores both forms).

### 3. Back end: device compile, then CPU codegen

- `Delegate` interface: `lower(func::FuncOp)` is removed; it becomes
  `compile(ModuleOp deviceModule, WeightArena &, out)`. The NPU delegate
  checks each `@npu_<id>` (one activation argument, one claimable matmul on
  it, its result returned), runs `npu::claim` and encodes as before.
  - Deviation: `compile` returns the ids in the device's index order rather
    than whole delegate rows. The bytes in and out come from the
    declaration's types, which `pynqo-backend` reads itself, so a delegate
    cannot disagree with the CPU graph about them.
- `partition.cpp` is deleted. `backend/cpu/delegate_calls.cpp`
  (`lowerDelegateCalls`) turns each call to a `@__delegate_<id>` declaration
  into the byte-view sequence and `func.call @__delegate__(%id, %in, %out)`
  and erases the declarations. It reads only `pynqo.delegate_id` and the
  tensor types.
  - Deviation: it runs on tensors, once before the per-target clones,
    instead of as a pass after `one-shot-bufferize`. It emits exactly the
    sequence `partition.cpp` emitted (alloc, `materialize_in_destination`,
    call, restrict `to_tensor`), which bufferization already handles, while
    a call to an external function with tensor results would need its own
    bufferization rule. It is still CPU code generation: it knows the ABI,
    not the device. The copy of `%x` into a fresh buffer therefore stays.
- `pynqo-backend` checks: declarations are bodiless, private, one static
  tensor in and one out, ids `0..n-1`; every declaration has exactly one
  function with its id and type in its device's module, and nothing else in
  that module has an id; every device has a delegate in `--devices`.
  `--keep-ir` writes the CPU module after this step as `cpu.mlir` (it was
  `partitioned.mlir`).

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
  hook covers it). Plan 004 is not on `dev` yet; this is recorded for when
  it is committed.
- Plan 005: phase 2's "partition inlines the lowered body" is superseded
  (marked in plan 005).
- `docs/manual/sw/trace_src.md`: the export trace and the package format.

## Package layout

```text
package/
├── ir/model.mlir        M  CPU graph, @__delegate_<id> declarations, no device bodies
├── ir/delegates/npu.mlir A  one @npu_<id> per delegate
├── compile.json         M  delegates[].output_bytes = M·N·4; format pynqo-backend/2
├── manifest.json        M  format pynqo-package/3
├── programs.*           -  same programs (encoding unchanged)
├── weights.bin          -  unchanged
└── <cpu>/model.so       M  no K-plane reduce; calls __delegate__ only
```

## Scope and acceptance

The user decided that the delegate call goes with partitioning ("delegate
應該要跟 dispatch 一起"), so both parts drafted as phases land together in
#178, on top of its middle-end assignment and partitioning.

| Part | Scope | Acceptance |
| --- | --- | --- |
| Delegate result | `npu.matmul` returns `M×N`, `libnpu_rt` sums K planes (wrapping), claim requires a zero init, format bumps | SmolLM2 24 greedy steps and ResNet-18 outputs bit-identical to `dev` on the host sim path; same delegate count and ids; `programs.bin` byte-identical; `model.so` has no K-plane reduce; runtime tests for multi-chunk delegates (`K > tk`); board run of both demos |
| Delegate call in the middle end | declarations + `ir/delegates/<device>.mlir`; `Delegate::compile(ModuleOp)`; CPU delegate-call lowering; `partition.cpp` deleted | `ir/model.mlir` has no device ops or bodies; mismatched modules fail with a clear message; lint, sim and export tests pass; `export_stages.md`, `filetree.md`, plan 005 updated |

## Results (#177, PR #178)

Measured against packages exported from `origin/dev` at `f74ef0e`, both
shipped configs (`devices: [cpu, npu]`, `cpus: [arm-cortex-a9, native]`),
front end on Windows, both C++ tools built fresh in WSL against LLVM 22.

**Delegate table and programs.** `programs.bin`, `programs.txt`,
`programs.json` and `weights.bin` are byte-identical for SmolLM2 and
ResNet-18. `compile.json` differs only in `format` (`pynqo-backend/1` → `/2`)
and `delegates[].output_bytes`: ids, devices, indices and input bytes are the
same (422 and 21 delegates). Output bytes total 36,430,848 → 10,968,576
(SmolLM2, all 422 delegates had K > 256) and 33,525,568 → 9,938,848
(ResNet-18, 17 of 21 changed; the other 4 have K ≤ 256). Every other key is
identical.

**IR.** `ir/model.mlir` has 422 and 21 `@__delegate_<id>` declarations, one
call each, and no `linalg.matmul`, `npu.` op or `@npu_` symbol;
`ir/delegates/npu.mlir` has 422 and 21 `@npu_<id>` functions, one
`linalg.matmul` each. The CPU module the back end compiles has no
`linalg.reduce` left (dev: 422 and 21, one per delegate).

**CPU code** (`arm-cortex-a9`, `model.llvm.mlir`): `malloc` 2522 → 2100 and
168 → 147 (one per delegate fewer: the accumulator), `memcpy` 1489 → 1075
and 51 → 36, `memrefCopy` unchanged (242, 211). `model.so` 1,203,208 →
1,129,288 bytes (SmolLM2) and 105,036 → 101,292 (ResNet-18); x86_64 shrinks
likewise.

**Outputs.** Host runs (x86_64 `model.so`, libnpu_rt's ISA simulator, the
runtime's `sim` path): SmolLM2 chat prompt "What is the capital of France?",
24 greedy tokens, same ids and the same SHA-256 of the logits after each of
the 26 package calls; ResNet-18 logits for the six demo and gallery images,
same SHA-256. Bit-identical.

**Tests.** `make -C src/test lint sim`: lint clean, 16/16 testbenches pass.
Export tests (`test_export_middle_end`, `test_export_layering`,
`test_export_pytorch`): 33 pass, including the new declaration and device
module checks, `--max-k 8` (K = 16 and 32 in 2 and 4 chunks with unchanged
output bytes and no reduce) and a refused device module that misses a
subgraph. `unittest discover`: 134 pass on Windows, 4 skipped (Linux-only
libnpu_rt tests); in WSL the new `test_npu_rt` (one chunk, five chunks with a
short last one, two chunks over several row and column tiles, a guard row
past M x N) and the profiler's libnpu_rt test pass.

**Not run: the board.** libnpu_rt's C unpack changed (it now sums planes), so
both demos still need a PYNQ-Z1 run before this is called done, and the host
time of the sum is to be read from the profiler there.

## Risks and open questions

- **Runtime ABI change.** Old packages (planes) and new runtime do not mix;
  the `format` bump makes it an error instead of wrong numbers.
- **Host time.** Summing in `unpackC` replaces a separate compiled
  `linalg.reduce` over the same data; expected neutral or faster (one pass,
  smaller `out`). To be measured with the profiler on the board.
- **Group-wise scales.** `npu.matmul`'s old description kept the planes
  apart so the CPU could scale each K group (group-wise INT4 weights). No
  scheme does that today. If one is added, the scale belongs inside the
  delegate (the device module and libnpu_rt), not in CPU code.
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
