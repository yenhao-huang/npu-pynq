# Middle end and back end split plan

| Field | Value |
| --- | --- |
| Plan ID | 2026-1112-005 |
| Topic | Every graph-level step, device assignment and partitioning included, in `pynqo-middle-end`; `pynqo-backend` only generates code per device and per CPU target and applies the runtime ABI |
| Date | 2026-11-12 |
| Status | Phase 1 done (#176); phase 2 done (#177) |
| Based on | `origin/dev` at `95c35f7` |
| Related | 2026-1112-004 (kernel outlining; runs after this plan's fusion), 2026-1112-002 (kernel profiler) |
| Issues | #176 (phase 1), #177 (phase 2) |

## Goal

Split the export compiler the way IREE and TVM do:

- **`pynqo-middle-end`** does every graph-level step: our torch rewrites, the
  lowering to linalg, the target-agnostic cleanups and elementwise fusion, and
  the decision of which device runs each operation, and the partitioning that
  moves each device-bound operation into its own device function. Its output,
  the package's `ir/model.mlir`, is the final graph: CPU code plus calls to
  `@npu_<id>` functions.
- **`pynqo-backend`** only generates code and applies the runtime ABI: each
  device compiles its device functions and replaces their calls with
  `__delegate__(id, in, out)`, and each `--cpus` target compiles the rest.

Outputs, the numeric contract (`src/test/model/`), the package format
(`compile.json`, `programs.bin`, `weights.bin`, `model.so` entry points) and
the runtime are unchanged.

## Problem

Today the two tools divide the work as follows.

`src/export/middle-end/pynqo_middle_end.cpp` runs `FuseConvBatchNorm`,
`DecomposeIntMm` and `AnyDimToAnyDims` on the torch dialect, then
`torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline`,
and prints linalg on tensors (`--emit linalg`, or `--emit torch` to stop after
our rewrites).

`src/export/backend/pynqo_backend.cpp` then:

1. registers the delegates for `--devices` (`device_register.cpp`; the
   interface is `delegates/delegate.h`);
2. runs `partition()` (`partition.cpp`), in which each delegate both **claims**
   and **compiles**: `NpuDelegate::claims()` (`delegates/npu/lower.cpp`) runs
   the vendor pass `--npu-claim`
   (`src/vendors/npu_pynq/custom_dialect/lib/Claim.cpp`), which rewrites every
   INT8 x INT8 -> INT32 `linalg.matmul` on a constant weight into
   `npu.matmul {tk}` plus a `linalg.reduce` over its K planes;
   `NpuDelegate::compile()` packs the weights and encodes `programs.*`; then
   partition replaces each `npu.matmul` with
   `func.call @__delegate__(id, in, out)`;
3. externalizes the remaining weights into `weights.bin`;
4. for each `--cpus` target, clones the module and runs `codegen()`
   (`cpu/toolchain.cpp`), whose `pipeline()` begins with
   `canonicalize,cse,linalg-fuse-elementwise-ops` before
   `one-shot-bufferize` and the LLVM lowering;
5. writes `compile.json`.

Two consequences:

- **Graph-level work runs in the back end, and runs repeatedly.**
  `canonicalize`, `cse` and `linalg-fuse-elementwise-ops` depend on neither the
  device nor the CPU target, yet they run once per `--cpus` entry on a cloned
  module. Both shipped configs (`src/export/configs/smollm.yml`,
  `resnet18.yml`) list `cpus: [arm-cortex-a9, native]`, so every export fuses
  the whole graph twice. The middle end's output is not the graph the code
  generators actually compile.
- **The back end has two kinds of responsibility.** Deciding which operations
  the NPU runs (a pattern match over the graph) and generating NPU programs
  (packing, encoding) are entangled inside one `Delegate` call sequence in
  `partition.cpp`. The decision cannot be inspected in `ir/model.mlir`, and it
  cannot be made without linking the NPU code generator.

## Prior art

The two systems below were checked only through forum posts (TVM) and commit
messages (IREE), not against their source.

| System | Where the device is decided | What the per-device back end does |
| --- | --- | --- |
| TVM BYOC | Relay passes on the graph, before build: `AnnotateTarget` marks the ops an external codegen supports, `MergeCompilerRegions` merges adjacent marked regions, `PartitionGraph` outlines each region into a function tagged with its compiler | Partitioned functions go to the external codegen; everything else goes through TIR / LLVM ([RFC: an extended graph partitioning flow](https://discuss.tvm.apache.org/t/rfc-byoc-an-extended-graph-partitioning-flow/6028), [Bring your own codegen to TVM](https://discuss.tvm.apache.org/t/bring-your-own-codegen-to-tvm/4501)) |
| IREE | Operations carry `#hal.device.affinity<@device>`, with a module default in `stream.affinity.default`; Stream's `ScheduleExecutionPass` partitions by affinity and inserts the transfers | HAL target backends only compile executables ([#17987](https://opensecura.googlesource.com/3p/openxla/iree/+/d39c3c56682e006e842b32aa6f38c272f77c8f3c), [#24748](https://opensecura.googlesource.com/3p/openxla/iree/+/df5f71837e57f82435daa0a8678bdffb9d67bdac)) |

In both, the device decision is a graph-level annotation made before code
generation, and device annotation and partitioning happen before operator
fusion (TVM's `FuseOps` runs inside build; IREE forms dispatch regions after
affinities are set). TVM's `PartitionGraph` is the model for this plan's
partitioning: it outlines each annotated region into a function carrying a
`Compiler` attribute, and the external codegen receives those functions, not
loose operators.

**Trade-off.** The middle end becomes device-aware: like TVM's annotation
rules, it must know each device's claim pattern. It stays free of code
generation (no packing, no encoding, no ISA) and independent of the CPU
target.

## Design

### 1. Target pipeline

```text
pynqo-middle-end (graph level)                          --devices cpu[,npu]
  FuseConvBatchNorm, DecomposeIntMm, AnyDimToAnyDims     torch dialect (unchanged)
  torch-function-to-torch-backend-pipeline,
  torch-backend-to-linalg-on-tensors-backend-pipeline    (unchanged)
  pynqo-assign-devices                                   new: pynqo.device = "npu" on claimable ops
  pynqo-partition-devices                                new: each annotated op -> func @npu_<id>,
                                                         replaced by func.call @npu_<id>; ids fixed here
  canonicalize, cse, linalg-fuse-elementwise-ops         moved from cpu/toolchain.cpp; upstream pass,
                                                         calls are opaque so fusion never enters a device function
  -> ir/model.mlir                                       final graph: CPU code + calls to @npu_* functions
pynqo-backend (code generation and runtime ABI)         --devices, --cpus, NPU geometry
  check module pynqo.devices == --devices
  partition, per device: its @<device>_* functions, in delegate-id order
    NPU: lower each body (npu.matmul {tk} + linalg.reduce), pack weights,
         encode programs.*; inline the lowered body at its call site and
         replace the npu.matmul with func.call @__delegate__(id, in, out)
  externalize remaining weights -> weights.bin
  per --cpus target (cpu/toolchain.cpp):
    one-shot-bufferize ... LLVM dialect, llc, link -> <cpu>/model.so
  compile.json
```

### 2. Fusion moves to the middle end (phase 1)

`pynqo_middle_end.cpp` appends `canonicalize`, `cse` and
`linalg-fuse-elementwise-ops` after the lowering pipelines, only for
`--emit linalg`. The passes are added through their C++ constructors
(`createCanonicalizerPass`, `createCSEPass`,
`createLinalgElementwiseOpFusionPass`) so the middle end does not need to
register the whole linalg pass library for a pipeline string; its
`CMakeLists.txt` links `MLIRLinalgTransforms` if torch-mlir's libraries do not
already pull it in. (Phase 1: torch-mlir's libraries already pulled it in
transitively; it is now listed explicitly because the middle end calls its
constructor directly.)

`cpu/toolchain.cpp` `pipeline()` drops the three passes and starts at
`one-shot-bufferize`. The later `canonicalize` (after
`buffer-results-to-out-params`) stays: it cleans up bufferization, which is
per target.

Expected effect: identical delegate table and `programs.bin` (claims are made
on the same graph, see section 5), bit-identical model outputs, less backend
time per extra `--cpus` target.

Open point: the old leading `canonicalize,cse` also cleaned up what
`partition()` and `externalizeWeights()` insert (per-delegate
`arith.constant` ids and zero indices, `memref.view`, `memref.cast`,
`bufferization.to_tensor`). Phase 1 compares `memref.alloc` / `memref.copy`
counts in `model.llvm.mlir` (`--keep-ir`) with and without them. If they
differ, the back end keeps one `canonicalize,cse` over its own insertions, run
once in `pynqo_backend.cpp` after `externalizeWeights` and before the
per-target clones, not inside `pipeline()`.

**Answered in phase 1: no back-end cleanup is needed.** In `model.llvm.mlir`
(`--keep-ir`, arm-cortex-a9), where `memref.alloc` has become `llvm.call
@malloc` and `memref.copy` has become `llvm.call @memrefCopy` or
`llvm.intr.memcpy`, the counts without the leading `canonicalize,cse` equal
those with it (the `dev` back end):

| Model | `malloc` | `memrefCopy` | `llvm.intr.memcpy` |
| --- | --- | --- | --- |
| ResNet-18, before and after | 168 | 211 | 51 |
| SmolLM2, before and after | 2522 | 242 | 1489 |

The later `canonicalize` inside `pipeline()` (after
`buffer-results-to-out-params`) already folds what partition and
externalization insert. Every `model.so` is byte-identical to `dev`'s (see
section 5), so `pynqo_backend.cpp` gains no cleanup pass.

### 3. Device assignment and partitioning in the middle end (phase 2)

Two new middle-end passes run right after the lowering, before fusion.

**`pynqo-assign-devices{devices=cpu,npu}`.** For each device in `--devices`
order except `cpu` (which runs whatever nobody claims), it walks the module
and annotates every operation the device's matcher accepts and no earlier
device took, with `pynqo.device = "<device>"`. The back end's existing
first-claim-wins rule (`partition.cpp`, `taken`) moves here unchanged in
meaning.

**`pynqo-partition-devices`.** For each annotated operation it outlines the
operation, together with what it needs and nothing else uses, into a private
device function, and replaces it with a call:

- the function's arguments are the operation's non-constant inputs (the
  activation); its result is the operation's result;
- constant operands (the weight) and the accumulator initialization
  (`tensor.empty` + `linalg.fill` of the `outs` operand) are cloned into the
  body, so the signature is "activation in, result out";
- the function carries `pynqo.device = "npu"` and `pynqo.delegate_id = <id>`;
- ids are assigned here, globally, in the order today's `partition()` assigns
  them: by device in `--devices` order, then in module order. With one
  delegate this is module order.

```mlir
// ir/model.mlir, abridged
module attributes {pynqo.devices = ["cpu", "npu"]} {
  func.func @decode(...) {
    ...
    %y = func.call @npu_17(%x) : (tensor<8x576xi8>) -> tensor<8x1536xi32>
    ...
  }
  func.func private @npu_17(%x: tensor<8x576xi8>) -> tensor<8x1536xi32>
      attributes {pynqo.device = "npu", pynqo.delegate_id = 17 : i64} {
    %w   = arith.constant dense_resource<torch_tensor_576_1536_torch.int8> : tensor<576x1536xi8>
    %e   = tensor.empty() : tensor<8x1536xi32>
    %acc = linalg.fill ins(%c0_i32 : i32) outs(%e : tensor<8x1536xi32>) -> tensor<8x1536xi32>
    %y   = linalg.matmul ins(%x, %w : tensor<8x576xi8>, tensor<576x1536xi8>)
             outs(%acc : tensor<8x1536xi32>) -> tensor<8x1536xi32>
    return %y : tensor<8x1536xi32>
  }
}
```

Partitioning before fusion is what keeps device work out of fusion without a
custom pass: a `func.call` is opaque to `linalg-fuse-elementwise-ops`, so the
upstream pass can neither fuse into a device function nor reshape or fold its
contents.

**The contract between the tools** is the device functions and three
attributes:

- `pynqo.device` (a string) on each device function;
- `pynqo.delegate_id` (an integer) on each device function. The middle end
  fixes the ids; the back end must keep them. Partition checks that the ids it
  receives are `0..n-1` without gaps or repeats and writes the `compile.json`
  table from them unchanged;
- `pynqo.devices` (an array) on the module, so the back end can refuse a
  `model.mlir` produced for a different device list.

Nothing else crosses: the only NPU parameter the lowering uses,
`tk = min(K, max_k)`, is a hardware limit computed in the back end, which
already receives `--max-k`; the claim decision itself does not depend on
`max_k` today.

The attribute names have no registered dialect behind them; phase 2 confirms
that the back end parses them without `allowUnregisteredDialects`
(discardable attributes are expected to be accepted; if not, the names become
`pynqo_device` and `pynqo_delegate_id`).

### 4. The back end lowers device functions (phase 2)

Today one interface, used only by the back end:

```cpp
class Delegate {                                   // backend/delegates/delegate.h
  virtual std::string name() const = 0;
  virtual FailureOr<std::vector<Operation *>> claims(ModuleOp) = 0;   // match + rewrite
  virtual LogicalResult compile(const std::vector<std::pair<int64_t, Operation *>> &,
                                WeightArena &, const std::string &out) = 0;
};
```

After the split the matcher lives in the middle end and the back end works
per device function:

```cpp
// middle end: middle-end/linalg/transforms/AssignDevices.cpp
// A device's claim rule: pattern match only, no rewrite, no codegen.
struct DeviceMatcher {
  std::string name;                                // "npu"
  std::function<bool(Operation *)> matches;        // npu: npu::isClaimable
};

// back end: backend/delegates/delegate.h
class Delegate {
  virtual std::string name() const = 0;
  // Rewrite one device function's body into the form compile() reads and
  // return the operation the device runs (the NPU: the npu.matmul).
  virtual FailureOr<Operation *> lower(func::FuncOp fn) = 0;
  virtual LogicalResult compile(...) = 0;          // unchanged
};
```

`partition.cpp` shrinks to: find the device functions; check the ids; for each
registered delegate, `lower()` its functions and `compile()` the returned
operations with the given ids; then apply the runtime ABI. A device function
for a device that is not registered is an error.

Applying the ABI cannot be a plain "replace `func.call @npu_17` with
`__delegate__`": the NPU's lowered body is not all device work.
`npu::claim` produces `npu.matmul {tk}`, which runs on the NPU and writes one
INT32 partial per K plane, plus a `linalg.reduce` over the planes into the
accumulator, which runs on the CPU (`Claim.cpp`; `npu_rt`'s `unpack_c` returns
`ceil(K/tk)` planes). So partition inlines each lowered body at its call site,
erases the device function, and outlines the `npu.matmul` exactly as today
(`__delegate__(id, in, out)` over flattened `memref<?xi8>` buffers). The
reduce and the fill stay as CPU code next to the call. They are created after
fusion, which loses nothing: the reduce has a reduction iterator, so
elementwise fusion would not take it as a producer, and its input is the
delegate's buffer, not a linalg op.

The rule "a delegated operation has one non-constant operand and one result"
moves to the device function's signature: one tensor argument (the
activation), one tensor result, every constant captured in the body. The
middle end establishes it when it outlines; the back end checks it before
lowering.

Where the code lives, respecting `docs/rules/filetree.md` (`src/vendors/`
sits below `src/export/`; `custom_dialect/` owns the NPU's claim rule):

| Piece | Location | Library | Linked by |
| --- | --- | --- | --- |
| `npu::isClaimable(Operation *)`: `linalg.matmul`, no user-defined maps, two inputs, static 2-D i8 x i8 -> i32, right operand an `arith.constant` (the predicate in today's `Claim.cpp`) | `src/vendors/npu_pynq/custom_dialect/lib/Match.cpp`, `include/Npu/NpuMatch.h` | new `NpuMatch` (upstream linalg and arith only; no tablegen, no npu dialect) | `pynqo-middle-end`, and `NpuDialect` |
| `npu::claim(linalg::MatmulOp, maxK)`: the rewrite to `npu.matmul {tk}` + `linalg.reduce`; the `--npu-claim` pass keeps working standalone by calling `isClaimable` + `claim` | `custom_dialect/lib/Claim.cpp` | `NpuDialect` (unchanged target, now also links `NpuMatch`) | `pynqo-backend` |
| Pack, Encode, `writePrograms` | `custom_dialect/lib/Pack.cpp`, `Encode.cpp`, `NpuEncoder.cpp` | `NpuDialect` | `pynqo-backend` |
| Device list -> matchers; outlining into device functions | `src/export/middle-end/linalg/transforms/AssignDevices.cpp`, `PartitionDevices.cpp` | `pynqo-middle-end` | |
| Device list -> delegates | `src/export/backend/device_register.cpp` (as today) | `pynqo-backend` | |

The vendor never learns the attribute names: the back end's NPU delegate
receives device functions, and the middle end calls the predicate. The
attribute strings are spelled in both tools (separate CMake projects); a test
checks the spellings agree.

Linking caution: `NpuDialect` links the monolithic `MLIR` shared library,
while `pynqo-middle-end` links individual static MLIR libraries alongside
torch-mlir. `NpuMatch` must not link `MLIR`; it declares its upstream
component dependencies (`MLIRLinalgDialect`, `MLIRArithDialect`) or none,
and lets each consumer supply them. Mixing the two forms in one binary risks
duplicate registration at start-up. The middle end's CMake adds the vendor
folder with `add_subdirectory` and links `NpuMatch` only; the `NpuDialect`
target is defined there but never built for `--target pynqo-middle-end`.

### 5. Order of claiming and fusion (decided: claim and partition before fusion)

In phase 2 the order is fixed by construction: assignment and partitioning
run before fusion, and fusion sees device work only as opaque calls. This
removes the risk that fusion changes the claim set:
`linalg-fuse-elementwise-ops` also runs reshape propagation (folding
`tensor.collapse_shape` producers and `tensor.expand_shape` consumers by
expanding a linalg op's iteration space) and linalg constant folding. If, in
LLVM 22, the expansion patterns accept named `LinalgOp`s, a `linalg.matmul`
next to such a reshape (the usual shape of an LLM linear layer) would become
a rank-3 `linalg.generic` that `isClaimable` no longer matches; constant
folding could create new constant weights and add claims. *Unverified*:
whether LLVM 22's patterns do this was not checked against source for this
plan. With claims and partitioning first, it no longer matters. Both
prior-art systems also annotate and partition before fusing.

Phase 1 alone has the other order: it moves fusion into the middle end while
claims still happen in the back end, so its delegate-table check measures
claim-after-fusion (**measured in phase 1: no change; the fallback below was
not needed and phase 1 landed as written**):

- if the table and `programs.bin` are identical for SmolLM2 and ResNet-18,
  phase 1 lands as written and the measurement is recorded here;
- if they differ, the differing operations are recorded, and phase 1 lands
  the fallback instead: fusion stays in the back end but runs once in
  `pynqo_backend.cpp` after `partition()` and before the per-target clones
  (removing the repetition, not yet the misplacement). Phase 2 then moves it
  into the middle end behind partitioning.

### 6. `--devices` reaches the middle end

`src/export/pipeline.py` today passes `--devices` and `--cpus` only to the
back end (`export_package`, the `_linux(f"{tool} ... --devices ...")` call);
`middle_end()` takes only `tool` and `emit`. Phase 2 adds a `devices`
parameter to `middle_end()` (default `('cpu',)`, so tools and tests that call
it for a CPU-only graph are unchanged) and `export_package` passes
`config['devices']` to both tools. `pynqo-middle-end` gains
`--devices` (default `cpu`). The back end compares the module's
`pynqo.devices` with its own `--devices` and fails on a mismatch, so a stale
`model.mlir` cannot be compiled for another device set.

`ir/model.mlir` then records the assignment and the partition (every
`@npu_*` function and its id), which makes "why did this matmul not go to the
NPU" answerable by reading one file.

`check_scheme` (refuse an NPU package with a scheme the NPU cannot run) stays
in `pipeline.py` and is unaffected.

### 7. Relation to plan 2026-1112-004

Independent. Plan 004 outlines CPU kernels in the back end, after NPU code
generation, so the device functions are already inlined and their device work
is `__delegate__` calls when it runs. The fusion move belongs to this plan only; plan 004 leaves fusion where
it finds it. Once this plan
lands, 004's tensor-level option (A) runs at the start of the back end's CPU
path, after partition, once for all targets, on a graph the middle end has
already fused.

## Change layout

Files this plan touches (`A` added, `M` modified), with the phase in brackets.

```text
src/export/
├── pipeline.py                          M  [2] middle_end(devices=...); export_package passes config['devices'] to both tools
├── middle-end/
│   ├── CMakeLists.txt                   M  [1] link MLIRLinalgTransforms if needed;
│   │                                       [2] add_subdirectory(vendor custom_dialect), link NpuMatch only, new sources
│   ├── pynqo_middle_end.cpp             M  [1] append canonicalize, cse, fusion; [2] --devices, pynqo-assign-devices and
│   │                                       pynqo-partition-devices before fusion, pynqo.devices on the module
│   └── linalg/transforms/
│       ├── Passes.h                     A  [2] constructors for the two passes
│       ├── AssignDevices.cpp            A  [2] device list -> matchers; writes pynqo.device, first claim wins
│       └── PartitionDevices.cpp         A  [2] annotated op + its constants and fill -> func @npu_<id>; call in its place;
│                                           delegate ids assigned here
└── backend/
    ├── pynqo_backend.cpp                M  [1] step comments; [2] check pynqo.devices against --devices
    ├── partition.h                      M  [2] doc: partition lowers device functions instead of asking delegates to claim
    ├── partition.cpp                    M  [2] shrinks: find device functions, check ids, lower(), compile(), inline the
    │                                       lowered body, replace npu.matmul with __delegate__ as today
    ├── device_register.h                M  [2] comment only: delegates no longer claim
    ├── delegates/delegate.h             M  [2] claims(module) -> lower(func::FuncOp)
    ├── delegates/npu/lower.cpp          M  [2] lower(): npu::claim on the device function's matmul; compile() unchanged
    └── cpu/toolchain.cpp                M  [1] pipeline() drops canonicalize, cse, linalg-fuse-elementwise-ops
src/vendors/npu_pynq/custom_dialect/
├── CMakeLists.txt                       M  [2] NpuMatch static library; NpuDialect links it
├── include/Npu/NpuMatch.h               A  [2] npu::isClaimable
├── include/Npu/NpuPasses.h              M  [2] npu::claim(linalg::MatmulOp, maxK) beside createClaimPass
├── lib/Match.cpp                        A  [2] the predicate, moved out of Claim.cpp
└── lib/Claim.cpp                        M  [2] the rewrite as npu::claim; --npu-claim = isClaimable + claim
src/test/tests/
├── test_export_middle_end.py            M  [1] fusion happens in the middle end; [2] assignment with [cpu] and [cpu, npu],
│                                           @npu_* functions with ids and signatures, pynqo.devices on the module
└── test_export_layering.py              M  [2] pynqo-middle-end links NpuMatch and not NpuDialect;
                                            the pynqo.device spelling agrees between the tools
docs/rules/filetree.md                   M  [1] middle end: torch dialect -> fused linalg;
                                            [2] -> fused linalg with device functions; custom_dialect: NpuMatch linked by
                                            both tools, NpuDialect by pynqo-backend only
```

Unchanged: `src/runtime/`, `src/vendors/npu_pynq/spec/` and `driver/`,
`src/hw/`, `src/test/model/`, `src/export/frontend/`, the
`compile.json` schema, the `programs.bin` / `programs.txt` / `programs.json` /
`weights.bin` formats, the `model.so` entry points and C interface,
`package.cpp`, and the `--npu-claim` pass as a standalone tool.

Package layout:

```text
package/
├── ir/model.mlir     M  fused; with [cpu, npu], @npu_* device functions and their calls
├── compile.json      -  unchanged (same delegate table)
├── programs.*        -  unchanged (byte-identical)
├── weights.bin       -  unchanged
└── <cpu>/model.so    -  same symbols and outputs; bytes may differ (see design section 2)
```

## Phase 1 results (#176)

Measured on `origin/dev` at `f74ef0e` (before) and the phase 1 branch (after),
both shipped configs (`devices: [cpu, npu]`, `cpus: [arm-cortex-a9, native]`).
Front end on Windows (torch 2.11.0+cpu, torchao 0.18.0, transformers 5.21.1,
torch-mlir wheel); both C++ tools in WSL against LLVM 22 with clang-22.

**Claim after fusion.** `ir/torch.mlir` is byte-identical before and after;
`ir/model.mlir` differs (it is now fused). Against the `dev` packages,
`compile.json` (so the `delegates` table: 422 entries for SmolLM2, 21 for
ResNet-18), `programs.bin`, `programs.txt`, `programs.json`, `weights.bin`
and both `model.so` files are byte-identical. Fusion before the claim moved no
claimable `linalg.matmul`, so the reshape-propagation and constant-folding
risk of section 5 did not occur for these two models; the fallback (fusion
once in the back end after partition) was not needed. Phase 2 still fixes the
order by construction.

**Outputs.** Host runs (WSL, the x86_64 `model.so`, `libnpu_rt`'s ISA
simulator, the runtime's `sim` path): SmolLM2 chat prompt "What is the capital
of France?", 24 greedy tokens, same token ids and same SHA-256 of the logits
after each of the 26 package calls; ResNet-18 logits for the six demo and
gallery images, same SHA-256. Bit-identical, as the identical `model.so` bytes
imply. Not run on the board.

**Back-end time** (`pynqo-backend` on `ir/model.mlir`, `--keep-ir`, output on
the WSL file system, two runs each, seconds):

| Model | `--cpus` | before | after |
| --- | --- | --- | --- |
| ResNet-18 | arm-cortex-a9 | 1.50, 1.01 | 1.08, 0.93 |
| ResNet-18 | arm-cortex-a9,native | 1.70, 1.62 | 1.70, 1.73 |
| SmolLM2 | arm-cortex-a9 | 29.0, 29.4 | 28.0, 29.2 |
| SmolLM2 | arm-cortex-a9,native | 55.1, 54.7 | 53.4, 54.7 |

The saving is within run-to-run noise: the three passes are cheap next to
bufferization, the LLVM lowering and `llc`, which stay per target. The new
middle end took 6.4 s (SmolLM2) and 0.6 s (ResNet-18) on the same
`torch.mlir`. Phase 1 removes the repetition and puts the passes where they
belong; it is not a speed-up.

## Phase 2 results (#177)

Measured on `origin/dev` at `f74ef0e` (before) and phase 2 rebased on phase 1
(after), same configs and tools as phase 1.

**Delegate table and programs.** Against the `dev` packages, `compile.json`
(422 delegates for SmolLM2, 21 for ResNet-18, same ids and order),
`programs.bin`, `programs.txt`, `programs.json` and `weights.bin` are
byte-identical. `ir/model.mlir` has 422 and 21 `@npu_*` functions, one per
delegate, with matching `pynqo.delegate_id`s.

**Outputs.** Same host runs as phase 1: SmolLM2 token ids and the SHA-256 of
the logits after all 26 package calls, and ResNet-18 logits for the six
images, are bit-identical. Not run on the board.

**`model.so` bytes differ** (both targets, both models), as section 2 and the
risk list allow. Each device function carries its own accumulator init
(`tensor.empty` + `linalg.fill`). On `dev`, matmuls of one shape shared an
init (8 inits for SmolLM2's 422 matmuls, 6 for ResNet-18's 21) because the
back end's leading `cse` merged them; phase 1 removed that `cse`, so after
inlining each delegate keeps its own zeroed buffer. The shared buffer had to
be copied before every K-plane `linalg.reduce` wrote into it, so the CPU code
loses those copies: `memcpy` calls 1489 → 1075 (SmolLM2, −414 = 422 − 8) and
51 → 36 (ResNet-18, −15 = 21 − 6); `malloc` counts are unchanged. Restoring a
`cse` in the back end was tried and rejected: it restores ResNet-18's bytes
but not SmolLM2's, because after fusion it merges 58 further allocations.

## Phases

| Phase | Scope | Changes | Acceptance |
| --- | --- | --- | --- |
| 1 | Fusion moves to the middle end | `canonicalize, cse, linalg-fuse-elementwise-ops` appended to `pynqo-middle-end`, removed from `cpu/toolchain.cpp` `pipeline()`; the cleanup question of design section 2 answered; `filetree.md` middle-end line updated | For SmolLM2 and ResNet-18: `compile.json` `delegates` table and `programs.bin` byte-identical to packages exported from `dev` before the change (this is also the claim-after-fusion measurement of design section 5, recorded in this plan; on a mismatch the fallback of section 5 lands instead); SmolLM2 24 greedy steps and ResNet-18 classifier outputs bit-identical; `make -C src/test lint sim` and the export tests (`src/test/tests/test_export_*.py`) pass; back-end time recorded with `cpus: [arm-cortex-a9]` and with both targets, before and after (no target figure); claim order decided and marked as decided in this plan |
| 2 | Assignment and partitioning in the middle end | `NpuMatch` library and `npu::claim`; `pynqo-assign-devices` and `pynqo-partition-devices` in the middle end, ahead of the upstream fusion pass; `--devices` through `pipeline.py`; `Delegate::claims()` replaced by `lower(func::FuncOp)`; partition lowers device functions and keeps their ids | Same delegate table (ids and order) and byte-identical `programs.bin` as packages exported from `dev`; outputs bit-identical; fusion is the upstream `linalg-fuse-elementwise-ops`, with no pynqo fusion pass; `ir/model.mlir` shows one `@npu_*` function per NPU delegate, with matching ids; `pynqo-middle-end` does not link `NpuDialect` (CMake check in `test_export_layering.py`, and no `npu::encode` / `packB` symbols in the binary); a mismatched `--devices` between the tools fails the export with a clear message; `filetree.md` updated for the middle end and `custom_dialect/`; lint, sim and export tests pass |

Each phase is one issue and one PR to `dev`, per the Git rules.

## Risks and open questions

- **Claim pattern versus fusion.** Phase 1 runs fusion before the existing
  claim; reshape propagation could turn a claimable `linalg.matmul` into a
  `linalg.generic` and constant folding could create new constant weights.
  The delegate-table check catches either; design section 5 gives the
  fallback. In phase 2 device work is outlined before fusion, so the claim set
  cannot move.
- **Delegate ids fixed early.** The middle end now decides the ids that
  `compile.json`, `programs.*` and the runtime agree on. A back end that
  renumbered would silently mismatch the program table; partition checks the
  ids form `0..n-1` and uses them as given.
- **Device function signatures.** All shapes are static today, so a device
  function has a fixed tensor signature. A dynamic dimension would make the
  outlined function take `index` arguments or dynamic tensors; the NPU claim
  rule already requires static shapes, so such ops are not claimed.
- **Bufferization of device functions.** Private device functions never reach
  bufferization: in `pynqo_backend.cpp`, `partition()` (step 3) inlines them
  and replaces their device work with `__delegate__` before `codegen()`
  bufferizes (step 6), so `bufferize-function-boundaries` sees no call to a
  device function.
- **Attributes dropped on the way.** `canonicalize`, `cse` and fusion run
  after partitioning, on the caller; they do not rebuild `func.func` ops, and
  `cse` does not merge calls (`func.call` has unknown side effects), so the
  function attributes and the call count are stable. Phase 2 checks the number
  of `@npu_*` functions against the delegate table.
- **Delegate interface churn.** `claims()` becomes `lower(func::FuncOp)` and
  partition's claim loop moves to the middle end. Only one delegate exists today, so the
  churn is contained, but a new device now touches two places: a matcher in
  the middle end and a delegate in the back end. `filetree.md`'s "a new device
  is a delegate under `backend/delegates/` and its vendor folder" changes
  accordingly.
- **The middle end now needs `--devices`.** A `model.mlir` is valid only for
  the device list it was assigned for. The module attribute and the back end's
  check turn a mismatch into an error; the default `cpu` keeps hand runs and
  CPU-only tests working, and `pipeline.py` always passes the config's list.
- **Two MLIR link styles in one binary.** `NpuMatch` must not drag the
  monolithic `MLIR` library into `pynqo-middle-end` (design section 4). The
  phase 2 build is checked on a clean build directory.
- **Middle-end build time.** `pynqo-middle-end` builds torch-mlir from source
  (memory-bound; `pipeline.py` keeps the job count low). Adding the vendor
  folder adds one small library; the tablegen step for `NpuDialect` is defined
  but not built for `--target pynqo-middle-end`.
- **Tests that assume the old split.** `test_export_middle_end.py` calls
  `pipeline.middle_end()` without devices and matches
  `linalg\.matmul ins\(.*xi8>, tensor<\S*xi8>\) outs\(.*xi32>\)` in
  `LoweringTest.test_w8a8_contraction_stays_an_int8_matmul`; with the default
  `('cpu',)` nothing is assigned or partitioned and the regex still holds. With
  `[cpu, npu]` the matmul moves into an `@npu_*` function, unchanged in form,
  so the regex still matches there, but a test asserting it sits in `@decode`
  or `@forward` would not. Its other assertions (named `linalg.conv_2d_nchw_fchw` survives, no
  `torch.` ops) hold after fusion (confirmed in phase 1, which also adds
  `test_elementwise_chain_is_fused`). There is
  no back-end test file today (`test_export_layering.py` covers front-end
  imports, `test_export_pytorch.py` the front end), so the byte-identical
  package comparison in the phases is done by hand against `dev` packages and
  recorded in the PR.
- **`model.so` bytes.** Moving fusion and possibly the leading cleanup can
  change generated code without changing results. Acceptance is on outputs,
  the delegate table and `programs.bin`, not on `model.so` bytes; a measurable
  slowdown on the board is investigated before phase 1 lands.

## Not in scope

- Kernel outlining and the kernel table: plan 2026-1112-004.
- The kernel profiler: plan 2026-1112-002.
- More than one NPU, new device types, or a cost-based assignment (the
  assignment is the existing first-claim-wins pattern rule).
- Data-driven schedules, a runtime interpreter or a memory planner.
- Merging adjacent claimed operations into larger regions (TVM's
  `MergeCompilerRegions`): each claimed contraction stays its own
  `__delegate__(id)`.
- Any change to `src/hw/`, the NPU ISA, `src/test/model/`, the package
  formats or the runtime.
- The deprecated JIT path under `src/deprecate/jit/`.
