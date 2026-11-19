# Kernel-based execution plan

| Field | Value |
| --- | --- |
| Plan ID | 2026-1112-004 |
| Topic | Kernel outlining in `pynqo-backend`: a kernel library plus a compiled schedule instead of whole-graph `model.so` |
| Date | 2026-11-12 |
| Status | Draft |
| Based on | `origin/dev` at `95c35f7` |
| Related | 2026-1112-002 (kernel profiler, L1 consumes the kernel table), 2026-1112-005 (middle-end/backend split: fusion and device assignment move to the middle end), #157 (asynchronous delegates) |
| Issues | none yet |

## Goal

Give the CPU side of a `pynqo-package/2` package explicit kernel boundaries.
Export produces:

1. a **kernel library**: one function per CPU kernel, plus the existing
   `__delegate__(id)` calls for NPU work;
2. a **schedule**: the entry points (`decode`, `prefill`, or a classifier's
   entry) reduced to a sequence of kernel calls.

At run time the schedule calls the kernels one by one. This is the
"host/device split" or "kernel-based execution" that IREE, TVM and
AOTInductor use. Outputs, the numeric contract (`src/test/model/`) and the
runtime ABI of the entry points are unchanged.

Outlining becomes **the only CPU code path**: it is always on, there is no
export flag to turn it off, and the whole-graph build is removed once phase 2
lands. Packages exported before the change still load, since the runtime
calls the same entry points.

## Problem

`src/export/backend/cpu/toolchain.cpp` compiles the whole module at once:
`pipeline()` runs `canonicalize, cse, linalg-fuse-elementwise-ops,
one-shot-bufferize{bufferize-function-boundaries, identity-layout-map},
buffer-results-to-out-params, buffer-deallocation-pipeline,
convert-bufferization-to-memref, convert-linalg-to-loops`, then the LLVM
conversions, and `llc -O3` builds `model.so`. Every linalg op becomes loops
inside its entry function. Nothing marks where one kernel ends and the next
begins. Consequences:

- **No per-kernel profiling.** On SmolLM2 the profiler attributes 33.7 s of a
  41.1 s run (82%) to "model.so and dispatch" with no further breakdown.
- **No per-kernel debugging.** An intermediate cannot be dumped and compared
  with the PyTorch reference at a kernel boundary, because there is none.
- **No kernel replacement.** A hand-written NEON kernel, or moving one more op
  to the NPU, means changing the lowering for the whole graph.
- **No runtime scheduling.** CPU work cannot overlap the NPU. `npu_rt` busy-polls
  `ISA_STATUS` (about 6,900 polls per run) while the CPU has nothing else to
  run.
- **Repeated layers emit separate code.** Thirty identical decoder layers
  produce thirty copies of the same loops.

The NPU side already has boundaries: `src/export/backend/partition.cpp`
replaces each claimed contraction with `func.call @__delegate__(id, in, out)`,
which `src/runtime/delegate/dispatch.c` forwards to `npu_rt_execute`
(`src/vendors/npu_pynq/driver/npu_rt/execute.cpp`) with that device's program
index. The CPU side is what lacks them.

## Prior art

Claims marked *(unverified)* were not checked against source for this plan.

| System | Where kernels are formed | Kernel library | Schedule | Notes |
| --- | --- | --- | --- | --- |
| IREE | Dispatch regions formed and outlined at tensor level, after fusion and before bufferization (Flow / DispatchCreation passes; names have moved between releases) | Per-target executable; a library exposes `exports.ptrs[]` | VM bytecode in the `.vmfb` | The runtime wraps each export call (`issue_call`); checked in `runtime/src/iree/hal/local/loaders/embedded_elf_loader.c` and `executable_library_util.c` |
| TVM | `FuseOps` groups ops into primitive functions, each lowered on its own | Compiled operator library | Graph executor JSON or Relay/Relax VM | Not MLIR-based |
| AOTInductor | Inductor generates one kernel per fused group | Generated kernels in the `.so` | A generated C++ wrapper that calls the kernels in order | Closest to this plan: the schedule stays compiled code |
| LiteRT / ExecuTorch | No kernel generation for CPU: the schedule names ops | CPU kernels prebuilt in the runtime, picked by an op resolver / kernel registry | Op list in the flatbuffer / `.pte` | NPU partitions become one opaque vendor binary per subgraph *(as described in public docs; not checked in source)* |

Our NPU path is already finer than the LiteRT/ExecuTorch vendor model: each
claimed matmul is its own `__delegate__(id)` call and program id, not one
blob per subgraph.

Terminology: **kernel outlining**, the inverse of inlining, as in MLIR's
`gpu-kernel-outlining` (which moves a `gpu.launch` body into a `gpu.func`).

## Design

### 1. Provenance: FX locations kept through the pipeline

No new pass. `torch.mlir` carries no locations today only because the front
end writes it with `operation.print(file=f)`
(`src/export/frontend/to_mlir/__init__.py`), which leaves them out, and
pynqo-middle-end re-parses that text. The front end prints with
`enable_debug_info=True` instead, and the middle end and backend print their
IR files the same way, so whatever the importer attached survives every
hand-off.

Expected path, *unverified*: `FxImporter`
(`src/export/frontend/to_mlir/torch_mlir/to_torch_dialect.py`) gives each
`torch.aten.*` op a location from its FX node (the node name, and the Python
stack trace that points at the `nn.Module` source line); the torch-to-linalg
patterns create their ops with the matched op's location; and
`linalg-fuse-elementwise-ops` merges the locations of fused producers and
consumers into a `FusedLoc`. A post-fusion kernel then lists every ATen op
(FX node) it was built from. Phase 1 checks all three. If `FxImporter`
attaches nothing useful, kernels are reported by kind and shape only
(`aten: []`); a tagging pass is not part of this plan. A pattern that drops
locations is fixed in our torch-mlir patch
(`src/export/middle-end/patch_torch_mlir.cmake`) only if phase 1 shows the
loss matters.

Locations do not change generated code: the backend emits no debug info.

### 2. Pass `pynqo-outline-kernels`

A new pass in the backend, `src/export/backend/outline_kernels.cpp`. Where
pynqo-backend runs it depends on the position chosen in phase 1 (section 3):
once in `pynqo_backend.cpp` right after partition (option A), or inside each
target's `pipeline()` in `cpu/toolchain.cpp` (option B). Either way it runs
after partition, so claimed ops are already `__delegate__` calls and never
land inside a kernel.

**Kernels.** Each `linalg.*` op (`generic`, `matmul`, `batch_matmul`,
`transpose`, `fill`, and any other named op that survives) and each
`memref.copy`. Not kernels: `memref.alloc`, `memref.dealloc`, `memref.subview`,
`memref.expand_shape`, `memref.collapse_shape`, `memref.view`, casts and
constants; these are buffer bookkeeping and stay in the schedule. `__delegate__`
calls are already calls and are left alone.

**Rewrite.** For each kernel op:

- create `func.func private @k<id>(...) attributes {passthrough = ["noinline"]}`,
  whose arguments are the op's buffer operands with their exact types (strided
  layouts included) plus any captured non-constant scalar;
- move the op into it unchanged; clone captured scalar constants
  (`arith.constant`) into the body instead of passing them;
- replace the op with `func.call @k<id>(...)`;
- copy the op's location to the function and the call, so the kernel keeps
  its ATen members.

Ids are global across the module, in schedule order, independent of the
delegate ids in `compile.json` `delegates`.

Before (option B position, memref level, one entry, abridged):

```mlir
func.func @decode(%ws: memref<?xi8>, %x: memref<1x576xf32>, %out: memref<1x576xf32>) {
  %t = memref.alloc() {alignment = 64} : memref<1x576xf32>
  %cst = arith.constant 9.99999974E-6 : f32
  linalg.generic {...} ins(%x : memref<1x576xf32>) outs(%t : memref<1x576xf32>) {
  ^bb0(%a: f32, %o: f32):
    %s = arith.addf %a, %cst : f32
    %r = math.rsqrt %s : f32
    linalg.yield %r : f32
  } loc(fused["aten.add#41", "aten.rsqrt#42"])
  call @__delegate__(%c17, %in, %buf) : (i64, memref<?xi8>, memref<?xi8>) -> ()
  ...
}
```

After:

```mlir
func.func @decode(%ws: memref<?xi8>, %x: memref<1x576xf32>, %out: memref<1x576xf32>) {
  %t = memref.alloc() {alignment = 64} : memref<1x576xf32>
  call @k812(%x, %t) : (memref<1x576xf32>, memref<1x576xf32>) -> ()
  call @__delegate__(%c17, %in, %buf) : (i64, memref<?xi8>, memref<?xi8>) -> ()
  ...
}

func.func private @k812(%x: memref<1x576xf32>, %t: memref<1x576xf32>)
    attributes {passthrough = ["noinline"]} {
  %cst = arith.constant 9.99999974E-6 : f32
  linalg.generic {...} ins(%x : memref<1x576xf32>) outs(%t : memref<1x576xf32>) {...}
    loc(fused["aten.add#41", "aten.rsqrt#42"])
  return
}
```

**Why `noinline`.** Today the pipeline has no MLIR inliner, and `llc` runs the
code generator rather than the IR optimization pipeline, so nothing is
expected to inline these functions as built. The attribute keeps the boundary
if the build later gains an `opt -O3` or LTO step, which would inline small
single-use internal functions and erase it. Phase 1 confirms that
`convert-func-to-llvm` carries `passthrough` to `llvm.func` and that
`model.ll` shows `noinline`, and checks the disassembly regardless.

### 3. Position in the pipeline (open; decided in phase 1)

| | Option A (IREE order) | Option B (memref level) |
| --- | --- | --- |
| Where | `pynqo_backend.cpp`, after partition and before the per-target `codegen` clones, at tensor level; fusion must already have run, so today `canonicalize, cse, linalg-fuse-elementwise-ops` run ahead of the pass here, and after plan 2026-1112-005 they come from the middle end | after `convert-bufferization-to-memref`, before `convert-linalg-to-loops` |
| Kernel signature | tensors in, tensor results; bufferization turns them into memrefs through `bufferize-function-boundaries` and `buffer-results-to-out-params` | memrefs, already final |
| What outlining does | changes the program bufferization sees; runs once, so kernel ids are identical across targets by construction | only moves ops; every allocation and copy is already fixed; runs once per target |
| Risk | the boundary may add allocations or copies: `function-boundary-type-conversion=identity-layout-map` applies to every function, so a strided slice passed to a private kernel may be copied to an identity layout, and in-place analysis across calls may be more conservative | signatures carry strided layouts and many descriptor arguments; no fusion or bufferization can see across the boundary, which is fine because both already ran |

The MLIR Bufferization documentation does not say whether private-function
boundaries add allocations or copies. Phase 1 therefore measures, for both
options on SmolLM2: `memref.alloc` and `memref.copy` counts in
`model.llvm.mlir` (`--keep-ir`), and decode time on the board, against the
unoutlined build. Default expectation: B, unless A shows no extra buffers,
since B cannot change buffer behaviour by construction.

### 4. Kernel table in `compile.json`

`pynqo_backend.cpp` (step 7) writes a `kernels` array beside `delegates`:

```json
{"id": 812, "entry": "decode", "kind": "linalg.generic", "device": "cpu",
 "aten": ["add_7", "rsqrt_2"],
 "operands": [{"shape": [1, 576], "dtype": "f32", "role": "in"},
              {"shape": [1, 576], "dtype": "f32", "role": "out"}]}
```

- `aten` comes from the kernel's location (`FusedLoc` members, or one name):
  the FX node names, with the source line when the importer provides it; the
  exact form is settled in phase 1;
- NPU work appears as `{"kind": "delegate", "device": "npu", "delegate": 17,
  ...}`, so the table lists every unit the schedule runs, in order;
- the table is written once per package and all `--cpus` libraries share it.
  With option A the ids are assigned once, before the per-target clones; with
  option B each target outlines its own clone, ids depend only on op order,
  and phase 2 checks that they match across targets.

The kernel profiler (plan 2026-1112-002, L1) reads this table for its hook
points and static cost data (op and byte counts are computed from `kind` and
`operands`).

### 5. The schedule stays compiled code

`decode` and `prefill` remain C-interface functions in `model.so`, now a
sequence of allocations, views and calls. The runtime (`src/runtime/`) calls
them exactly as today. This is the AOTInductor model, not IREE's VM.

A data-driven schedule (an op list in the package, executed by a runtime
interpreter) is a later option. It would need: a public C ABI per kernel, a
memory planner that assigns intermediate buffers (today done by bufferization
and `hoist-static-allocs`), a serialized schedule format and its loader, an
interpreter loop in the runtime, and tests for both. It buys dynamic
scheduling and kernel swap without re-linking; nothing in this plan requires
it.

### 6. Later uses (each its own phase and issue, not committed here)

- **Kernel dedupe.** Identical layers produce structurally equal kernels
  (weights arrive through `ws` views, not in the body). Hash with
  `OperationEquivalence`, keep one function, many call sites. Profiling then
  needs a call-site id argument (or a per-site wrapper) to keep per-layer
  times.
- **Intermediate dumps.** A debug build calls a dump hook after selected
  kernels; a host script compares each against the PyTorch reference of its
  ATen members.
- **Kernel replacement.** Link a hand-written NEON function under the name
  `@k<id>` (or a stable name keyed by kind and shape) in place of the
  generated one; or claim one more op for the NPU.
- **CPU/NPU overlap.** With an asynchronous `__delegate__` (start, then wait),
  the schedule can run independent CPU kernels during the PL run instead of
  polling. Relates to #157.
- **Profiling without re-export.** Every package is outlined, so the
  profiler's hooks (plan 2026-1112-002) could be compiled into every package
  and gated by a runtime flag; that choice belongs to plan 002.

## Change layout

Files this plan touches (`A` added, `M` modified). Phase 1 builds a prototype
of the same files on its spike branch; phase 2 lands them.

```text
src/export/
├── pipeline.py                         -  unchanged: outlining has no switch
├── frontend/to_mlir/__init__.py        M  print torch.mlir with enable_debug_info=True (keeps FX locations)
├── middle-end/
│   ├── pynqo_middle_end.cpp            M  print its output with debug info
│   └── patch_torch_mlir.cmake          M  only if phase 1 finds location loss that matters
└── backend/
    ├── CMakeLists.txt                  M  add outline_kernels.cpp
    ├── outline_kernels.h               A  createOutlineKernelsPass, KernelRecord
    ├── outline_kernels.cpp             A  pynqo-outline-kernels; collects the kernel records
    ├── pynqo_backend.cpp               M  "kernels" array in compile.json; runs the pass after
    │                                      partition if phase 1 picks option A
    └── cpu/
        ├── toolchain.h                 M  codegen() returns the kernel records (option B)
        └── toolchain.cpp               M  pipeline() split around the pass if option B (with option A,
                                           fusion runs before it in pynqo_backend.cpp until plan 005
                                           moves it); the whole-graph pipeline is removed
src/test/tests/
└── test_export_outline_kernels.py      A  pass checks on a small module; logits against the pre-change package
docs/rules/filetree.md                  M  name outline_kernels
```

Unchanged: `src/runtime/` (it calls `decode`/`prefill` as today and ignores
`kernels`), `src/vendors/`, `src/hw/`, `src/test/model/`, and the
`programs.bin`/`weights.bin` formats.

Package layout:

```text
package/
├── compile.json      M  new "kernels" array beside "delegates"
├── manifest.json     -  unchanged
├── <cpu>/model.so    M  same exported symbols; internally decode/prefill call k0..kN
├── programs.*        -  unchanged
└── weights.bin       -  unchanged
```

`model.so` before and after:

```text
before                          after
decode()                        decode()                 schedule
  loops of kernel 0               call k0                 (allocs, views, calls)
  loops of kernel 1               call k1
  __delegate__(17)                __delegate__(17)
  loops of kernel 2               call k2
  ...                             ...
                                k0(), k1(), k2(), ...    kernel library (internal, noinline)
```

## Phases

| Phase | Scope | Changes | Acceptance |
| --- | --- | --- | --- |
| 1 | Spike | Front end prints with debug info; a prototype `pynqo-outline-kernels` at both positions; run on a small graph (linear, add, softmax, one NPU-claimed `_int_mm`) and on SmolLM2 IR | What `FxImporter` attaches is recorded, and whether it reaches the post-fusion kernels (or the dropping patterns are listed); A and B compared on `memref.alloc`/`memref.copy` counts and decode time, and one chosen; logits bit-identical to the unoutlined package; disassembly of `model.so` shows the kernels as separate, called functions; NPU claim count unchanged; kernel count recorded (for scale: the current SmolLM2 middle-end output, before fusion, holds about 7,157 `linalg.generic`, 422 `linalg.matmul`, 120 `linalg.batch_matmul`, 300 `linalg.transpose` and 28 `linalg.fill` across both entries) |
| 2 | Outlining as the only path | The pass runs unconditionally at the chosen position; the whole-graph path is deleted, with no fallback flag; `kernels` table in `compile.json` | FileCheck-style tests for the pass (kernel selection, captured constants, strided operands, locations kept); SmolLM2 (24 greedy steps) and ResNet-18 packages give bit-identical outputs to packages exported from `dev` before the change; kernel ids identical across `--cpus` targets; `make -C src/test lint sim` unaffected; board decode time within 2% of the pre-change package; above that, the pass is fixed (for example by leaving trivial kernels inline) before it lands |
| later | Uses from section 6 | Dedupe, dumps, kernel replacement, CPU/NPU overlap | Each defined in its own issue |

Each phase is one issue and one PR to `dev`, per the Git rules.

## Risks and open questions

- **Locations missing or dropped.** `FxImporter` may attach less than
  expected, and torch-mlir patterns or `linalg-fuse-elementwise-ops` may not
  keep or fuse locations. Phase 1 measures how many kernels end with an empty
  `aten` list; empty lists cost report detail, not correctness.
- **Outlining cost.** Each call passes memref descriptors as scalar arguments
  (`3 + 2 × rank` words per memref on ARMv7) and blocks cross-kernel
  optimization in `llc` (scheduling, register reuse at kernel edges). Small
  elementwise kernels may pay noticeably; the < 2% target is measured, not
  assumed. There is no fallback flag, so a cost above target blocks phase 2
  until the pass is fixed.
- **LLVM ordering.** Kernels and delegates touch memory through pointers
  passed as arguments, so call order is preserved by ordinary call semantics;
  confirmed by the bit-identical logits check rather than argued.
- **Strided memref types in private signatures.** Subview operands carry
  `strided<[...], offset: ?>` layouts; the private signature must keep them
  (option B) or bufferization may force copies (option A).
- **Index bitwidth.** Kernel signatures lower with the target's
  `index-bitwidth` like everything else; with option B the per-target clones
  in `pynqo_backend.cpp` must not change kernel ids.
- **Cross-compilation.** ARMv7 cross builds (`llc -mtriple`, `ld.lld`, the
  freestanding `memrefCopy` in `toolchain.cpp`) are unchanged; private kernel
  functions get internal linkage and need no new symbols or libraries.
- **`noinline` propagation.** If `convert-func-to-llvm` does not forward
  `passthrough`, the pass sets the attribute on the `llvm.func` after
  conversion instead.
- **Prefill padding and dynamic shapes.** All shapes are static today; a
  dynamic dimension would add `index` arguments to kernels, which the pass
  must handle as captured scalars.

## Not in scope

- A data-driven schedule, interpreter or memory planner (section 5).
- Changing fusion: kernels are whatever `linalg-fuse-elementwise-ops` leaves.
- Tiling, vectorization or NEON code generation for kernels.
- The profiler itself (hooks, clocks, reports): plan 2026-1112-002.
- Asynchronous delegates and CPU/NPU overlap (#157).
- Any change to `src/hw/`, the NPU ISA, `programs.bin`/`weights.bin` formats,
  or the numeric contract in `src/test/model/`.
- The deprecated JIT path under `src/deprecate/jit/`.
