## Why

Phase 1 executes bounded signed-INT8 matrix jobs, but a quantized ResNet-18
cannot yet be represented, tiled, exported, or executed without ad hoc host
code and potentially incorrect K-slice accumulation. Phase 2A establishes the
operator, memory, package, and runtime contracts required for Issue #6 while
preserving the Phase 0 numeric and ABI guarantees.

## What Changes

- Add a validated quantized graph model for the ResNet-18 operator subset:
  convolution, residual add, activation, pooling, flatten, and fully connected.
- Lower convolution and fully connected work to bounded matrix jobs, including
  edge tiles and K slices, without materializing a complete im2col tensor in
  DDR.
- Require an exporter-time accumulator-safety proof before K-sliced partial
  sums may be combined, so execution remains equivalent to ordered Phase 0
  signed INT32 accumulation.
- Add deterministic memory planning with bounded reusable scratch buffers and
  explicit tensor lifetimes.
- Add a deterministic, versioned export package containing commands, packed
  weights, quantization parameters, tensor metadata, integrity hashes, and the
  required hardware capabilities.
- Add a model runtime that validates packages before execution, dispatches all
  matrix work through the public Phase 1 runtime, executes defined host-side
  elementwise operators, enforces one model deadline, and returns owned output
  plus accounting metadata.
- Add focused golden-model, exporter, package-validation, and fake-runtime
  integration tests. Full ResNet-18 accuracy, performance, synthesis, timing,
  and physical-board acceptance remain Phase 2B Issue #7.

## Capabilities

### New Capabilities

- `quantized-resnet-operators`: Defines accepted tensor layouts, quantization,
  and bit-accurate behavior for the Phase 2A ResNet operator subset.
- `resnet-matrix-lowering`: Defines bounded convolution/fully-connected
  lowering, edge tiling, certified K slicing, deadlines, and result assembly.
- `resnet-memory-planning`: Defines deterministic tensor lifetimes, reusable
  scratch allocation, and the prohibition on full-graph/full-im2col DDR
  materialization.
- `npu-model-package`: Defines the deterministic versioned commands, weights,
  tensor metadata, capability requirements, and integrity validation emitted by
  the exporter.
- `npu-model-runtime`: Defines package preflight, public-runtime dispatch,
  host-operator execution, failures, repeatability, and execution metrics.

### Modified Capabilities

None. The Phase 0 numeric and ABI contracts and the Phase 1 matrix runtime
remain authoritative and are consumed without changing their requirements.

## Impact

- New production modules under `src/export/` and `src/runtime/`.
- The shared numeric model moves from `src/test/model/` to a production-visible
  location only if implementation dependency analysis requires it; existing
  imports remain compatible.
- New focused Python tests under `src/test/tests/` and deterministic fixtures
  under repository-approved test paths.
- No bitstream, Vivado project, generated model, dataset, or board output is
  committed. Phase 2A does not claim physical-board performance or end-to-end
  ResNet-18 accuracy evidence.
