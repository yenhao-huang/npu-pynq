## Why

Phase 2A can export only an already constructed internal `QuantizedGraph`, so
the repository cannot consume a real pretrained ResNet-18 and its reduced
fixtures do not exercise the production 224x224x3 input path. Issue #47 must
close that gap before Issue #7 can make honest model or board claims.

## What Changes

- Pin one official public TorchVision ResNet-18 weight revision and verify its
  digest when downloading into the example's gitignored model workspace.
- Add a deterministic host-only adapter that folds BatchNorm, calibrates a
  signed-INT8 graph with residual-compatible tensor scales, converts OIHW
  weights to HWIO, and calls the existing Phase 2A exporter.
- Produce provenance metadata and a small content-addressed real-model
  validation input/reference bundle without representing it as ImageNet
  accuracy evidence.
- Add an `examples/resnet18/` workflow that downloads, converts, verifies, and
  packages only completed model outputs from `examples/resnet18/model/`.
- Distinguish reduced-fixture, real-model host, and physical-board evidence in
  APIs, tests, and documentation.

## Capabilities

### New Capabilities

- `pretrained-resnet18-import`: Pinned source acquisition, deterministic
  TorchVision-to-`QuantizedGraph` conversion, provenance, and real-input
  agreement requirements.
- `resnet18-model-workspace`: Gitignored example model workspace, package
  readiness checks, commands, and evidence-level separation.

### Modified Capabilities

None. The Phase 0 numeric contract, Phase 1 ABI/RTL, and Phase 2A serialized
package format remain unchanged.

## Impact

- Affected paths: `src/export/`, `src/test/tests/`, `examples/resnet18/`,
  `.gitignore`, `docs/rules/filetree.md`, and this OpenSpec change.
- Host conversion adds explicitly pinned PyTorch/TorchVision dependencies;
  exported packages and PYNQ runtime remain NumPy-only.
- Downloaded weights, calibration inputs, converted packages, Vivado outputs,
  and evidence stay untracked.
- No RTL, register map, overlay topology, or board-network change is intended.
