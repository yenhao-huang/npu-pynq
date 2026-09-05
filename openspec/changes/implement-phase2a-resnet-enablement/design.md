## Context

Phase 1 exposes a physical signed-INT8 matrix job with runtime-discovered
MAX_M, MAX_N, and MAX_K. Each job resets its accumulators and applies the Phase
0 saturating INT32 rule after every MAC. Phase 1C tiles logical M and N but
correctly refuses K tiling because saturating addition is not associative.

ResNet-18 requires batch-one NHWC convolution with flattened K values larger
than the physical maximum, live residual branches, pooling, and a final fully
connected layer. The repository currently has no production-visible graph IR,
export directory implementation, model package, or model runtime. NumPy is
already required by the Phase 1 runtime; adding a framework dependency is not
necessary to define or test Phase 2A.

## Goals / Non-Goals

**Goals:**

- Freeze one small, serializable, framework-neutral quantized graph contract.
- Prove when K slicing is equivalent to the existing ordered Phase 0 contract.
- Keep im2col scratch bounded by physical limits and reuse dead tensor storage.
- Make exporter output byte deterministic and reject malformed packages before
  physical execution.
- Exercise the complete operator stack against a fake public Phase 1 runtime
  and bit-accurate references.

**Non-Goals:**

- Import arbitrary PyTorch or ONNX graphs; an adapter can translate into the
  frozen graph contract in a later independent change.
- Change constraints or the physical array dimensions.
- Support batch sizes other than one, grouped/depthwise/dilated convolution, or
  residual branches with different quantization.
- Claim ImageNet accuracy, full ResNet-18 latency, synthesis, timing closure, or
  physical-board acceptance; those are Issue #7 gates.

## Decisions

### 1. Promote shared numeric behavior and define immutable graph records

Production export and runtime code must not import from src/test. The Phase 0
integer primitives move to src/model/numeric.py, while src/test/model/numeric.py
remains a compatibility re-export during migration. Immutable records under
src/model/resnet.py represent tensors, quantization, packed constants, commands,
and a topologically ordered graph. Constructors validate scalar ranges and
local invariants; whole-graph validation checks names, producer/use order,
shapes, layouts, and operator-specific contracts.

Alternative considered: duplicate arithmetic in exporter and runtime. Rejected
because it creates three numeric contracts and makes differential evidence
ambiguous.

### 2. Use a conservative static proof to permit exact K slicing

For fixed signed-INT8 weights and optional signed-INT32 bias, the exporter
computes this per-output-channel bound:

    abs(bias[channel]) + 128 * sum(abs(weight[..., channel]))

Export fails when the bound exceeds INT32_MAX. A passing bound proves that no
possible signed-INT8 activation can overflow the channel accumulator, so every
physical K slice is also overflow-free. Physical signed INT32 partial results
can then be added in signed INT64, checked against the certificate, narrowed to
INT32, bias-added once, and requantized. Slices remain ordered to preserve a
simple mapping to the reference.

Alternative considered: unconditionally add K-slice results. Rejected because
per-MAC saturation is not associative. Extending RTL with accumulator preload
was also considered, but would require an ABI/overlay change and make Phase 2A
depend on new synthesis and board evidence.

### 3. Lower one spatial/output tile at a time

Convolution flattens output positions into M, kernel height/width/input channel
into K, and output channels into N. Iteration order is M tile, N tile, then K
slice. The lowerer creates only A[physical_M, physical_K] and
B[physical_K, physical_N], submits them through NPURuntime.run, and maintains
one INT64 accumulator tile. Input padding is generated as the input zero point.
Fully connected uses the same lowerer with M=1.

Host-side residual add, ReLU, max pool, global average pool, and flatten use
integer-only NumPy operations governed by the specs. These operations do not
benefit from the existing matrix RTL, and keeping them on the processing system
avoids inventing unsupported hardware capabilities.

Alternative considered: build a full im2col matrix for each convolution.
Rejected because its DDR footprint scales with logical output area and violates
Issue #6.

### 4. Plan a deterministic aligned activation arena

The planner computes half-open live intervals over command indices and uses a
deterministic first-fit allocator ordered by definition index then tensor id.
All offsets are 64-byte aligned. A free range can be reused only after the
previous tensor's final consumer. Residual inputs therefore remain live across
their branch. Scratch size is computed separately from physical M/N/K limits
and the largest logical N tile accumulator; it never contains full im2col.

The first implementation uses one host byte arena and typed NumPy views. It
does not preallocate PYNQ DMA buffers because NPURuntime owns that ABI boundary.

Alternative considered: allocate every tensor independently. Rejected because
it hides peak memory, prevents deterministic capacity checks, and retains all
intermediates.

### 5. Export a two-file canonical package

src/export/resnet.py writes NAME.npu.json and NAME.npu.bin through sibling
temporary files followed by atomic replacement after all validation succeeds.
The JSON uses UTF-8, sorted keys, compact separators, no NaN values, and one
trailing newline. It contains magic NPU_MODEL, format 1.0, required ABI and
capabilities, graph IO, tensors, ordered commands, the memory plan, payload
length, and SHA-256. The binary concatenates C-contiguous little-endian constants
in stable tensor-id order with 64-byte zero padding; every range is explicit.
No timestamp, source path, random identifier, pickle, or executable object is
stored.

Alternative considered: NPZ or pickle. Rejected because ZIP metadata can harm
byte determinism and pickle is executable and unsuitable as a trust boundary.

### 6. Separate package loading from model execution

src/runtime/model.py parses and completely validates both package files into an
immutable loaded model before accepting input. NPUModelRuntime then validates
the physical runtime limits, allocates its host arena, and executes commands
under one monotonic deadline. Matrix commands call only NPURuntime.run;
host operators never touch MMIO or DMA. Errors wrap the command id and tile
coordinates while preserving the underlying exception.

Metrics are accumulated from observable command and physical-call data. Cycle
sum uses physical result metadata when the Phase 1 runtime exposes it and is
otherwise explicitly unavailable rather than fabricated.

Alternative considered: stream-parse and execute commands immediately.
Rejected because a late malformed reference could cause partial hardware side
effects before package rejection.

### 7. Decompose Issue #6 into independently reviewable implementation units

Issue #6 remains the Phase 2A tracking/integration issue linked to this OpenSpec
change. Implementation is split into dependency-linked sub-issues for:

1. production numeric/graph contracts and golden operators;
2. deterministic memory planner and package exporter;
3. bounded matrix lowering with certified K slicing;
4. package loader/model runtime and integration tests.

Each sub-issue receives its own mandatory branch, worktree, tests, commit, and
PR to dev. The tracking change is updated as sub-issue evidence lands; Issue #6
closes only after all Phase 2A integration gates pass.

## Risks / Trade-offs

- [The overflow proof is conservative and may reject a numerically safe trained
  layer] -> Fail explicitly and leave a tighter proof or ABI accumulator-preload
  extension to a separate specified change.
- [2x2 physical tiling creates many DMA submissions and may miss performance
  goals] -> Preserve correctness and record exact call counts in Phase 2A;
  Issue #7 measures the board and can justify hardware optimization.
- [Host-side pooling and residual work can dominate latency] -> Make operator
  counts visible in metrics and keep package commands forward-compatible by
  rejecting, not guessing, unknown operator kinds.
- [Atomic replacement semantics differ across filesystems] -> Write both
  temporary files in the destination directory, fsync/close them, replace the
  payload first and manifest last; the manifest digest prevents mixed-pair use.
- [The full ResNet-18 graph may expose unsupported quantization alignment] ->
  Export fails at the exact residual command; Phase 2A does not silently insert
  floating-point or lossy rescaling.

## Migration Plan

1. Land the tracking OpenSpec and sub-issue graph without changing product code.
2. Land shared graph/numeric contracts and bit-accurate host references.
3. Land planner/exporter, then lowering, then runtime integration in dependency
   order; each PR targets dev and keeps existing Phase 0/1 tests green.
4. Run strict OpenSpec validation, focused Python tests, the full repository
   Python suite, RTL lint, and RTL simulation. RTL gates are regression evidence
   only because Phase 2A does not edit RTL.
5. Hand Issue #7 a deterministic package and host/fake-runtime evidence for
   full-model, synthesis/timing, and physical-board acceptance.

Rollback is commit/PR reversion in reverse dependency order. ABI v2 and its
matching Phase 1 overlay/runtime must be deployed together because hardware
requantization changes the register and stream contract.
