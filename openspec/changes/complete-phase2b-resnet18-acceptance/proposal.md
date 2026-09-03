## Why

Phase 2A can represent, export, and execute the ResNet operator subset, but it
does not yet prove that a complete ResNet-18 package, an approved quantized
reference corpus, and a PYNQ-Z1 execution agree end to end. Phase 2B adds the
reproducible acceptance boundary required by Issue #7 without committing
third-party weights, datasets, generated overlays, or board credentials.

## What Changes

- Add a strict acceptance-bundle manifest that binds a Phase 2A package,
  quantized input corpus, expected outputs, labels, reference identity, and
  file digests.
- Validate that the package has the canonical ResNet-18 basic-block topology,
  including eight residual additions and three projection shortcuts.
- Expose physical cycle telemetry through the public Phase 1 runtime and
  aggregate it through matrix lowering and model execution.
- Add deterministic host acceptance for exact tensor agreement, top-1
  accuracy, repeated inference, latency, throughput, physical jobs, MACs,
  operations, and cycles.
- Add a standalone `examples/resnet18/` package, board runner, deployment
  wrapper, and evidence schema whose final publication is transactional.
- Keep synthesis, timing, resource, and physical-board claims fail-closed when
  trusted evidence or the required external assets are unavailable.

## Capabilities

### New Capabilities

- `resnet18-acceptance-bundle`: Defines externally supplied, digest-bound
  model/corpus/reference assets and canonical ResNet-18 topology validation.
- `resnet18-acceptance-runner`: Defines deterministic host and PYNQ-Z1
  accuracy, repeatability, performance, and evidence behavior.
- `resnet18-board-delivery`: Defines standalone packaging, deployment,
  provenance, atomic promotion, rollback, and trusted board gates.

### Modified Capabilities

- `npu-model-runtime`: Adds physical cycle aggregation and optional immutable
  tensor capture needed for layer-level acceptance.

## Impact

- Affected code: `src/runtime/`, `src/model/`, `src/test/tests/`, and a new
  `examples/resnet18/` consumer following the existing example boundary.
- External inputs: an approved quantized ResNet-18 package and calibration or
  validation corpus remain untracked, hash-identified acceptance inputs.
- Hardware: no ABI or RTL change is intended; Phase 2B consumes the existing
  cycle-counter registers and Phase 1 overlay.
- Operations: final synthesis, timing, and board evidence require trusted
  Vivado and PYNQ-Z1 runners and cannot be inferred from host simulation.
