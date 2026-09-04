## 1. Contract and Test Baseline

- [x] 1.1 Add failing tests for pinned source metadata, safe atomic download, digest/redirect/overwrite rejection, and a gitignored `examples/resnet18/model/` workspace; verify the focused downloader suite fails for the missing implementation.
- [x] 1.2 Add a deterministic source-shaped ResNet-18 state fixture and failing tests for key/shape validation, BatchNorm folding, OIHW-to-HWIO conversion, signed quantization, residual scale equality, and byte-identical exports; verify the focused converter suite fails first.
- [x] 1.3 Add failing real-shape validation tests that require `(1, 224, 224, 3)` stem/first-stage agreement and distinct fixture/host/board evidence labels; verify they expose the current integration gap.

## 2. Source Acquisition and Workspace

- [x] 2.1 Add tracked immutable model-source metadata and a standard-library downloader under `examples/resnet18/`; verify successful local-fixture download and every fail-closed case without network access.
- [x] 2.2 Add `examples/resnet18/model/` to the repository filetree and ignore all generated contents except its placeholder; verify prepared model outputs leave `git status` clean.
- [x] 2.3 Download the pinned official checkpoint, record its full SHA-256 and byte length in metadata, rerun acquisition, and verify the local file matches both values.

## 3. Deterministic ResNet-18 Conversion

- [x] 3.1 Implement strict weight-only checkpoint loading and the fixed TorchVision ResNet-18 source schema; verify missing, extra, malformed, unsafe, and non-finite records fail before output.
- [x] 3.2 Implement deterministic float interpretation, BatchNorm folding, tensor-shape mapping, and calibration input generation; verify stem, four stages, pooling, and classifier float tensors match the pinned architecture.
- [x] 3.3 Implement signed per-channel weight/bias conversion, residual-compatible activation scale groups, Q1.31 parameters, and canonical `QuantizedGraph` construction; verify all Phase 2A graph/export safety checks pass.
- [x] 3.4 Add the conversion CLI and canonical provenance/evidence outputs; verify two identical conversions produce byte-identical NPU manifest, payload, and conversion metadata.

## 4. Real Model and Example Delivery

- [x] 4.1 Run the pinned real checkpoint through conversion and verify an independent integer reference agrees exactly at the stem, first residual stage, and final output for a deterministic full-shape input.
- [x] 4.2 Add package readiness and deterministic archive construction from `examples/resnet18/model/`; verify checkpoint-only, stale, substituted, incomplete, or unvalidated workspaces publish no archive.
- [x] 4.3 Add the output-free ResNet-18 notebook and human README with download, conversion, validation, Vivado, package, and board commands in executable order; verify notebook structure, links, paths, and expected markers.
- [x] 4.4 Ensure fixture, real-model host, and physical-board evidence names cannot alias; verify fake runtimes never emit a physical-board PASS marker.

## 5. Validation and Handoff

- [x] 5.1 Run focused downloader/converter/example tests and the full Python suite; record exact counts and commands in `STATE.md`.
- [x] 5.2 Run strict OpenSpec validation, `git diff --check`, secret/generated-artifact inspection, RTL lint, and RTL simulation; record passed, not-applicable, or blocked gates without substitution.
- [x] 5.3 Attempt the documented PYNQ-Z1 real-model path with matching trusted overlay artifacts; record physical PASS evidence or the exact external blocker without relabeling host results.
- [x] 5.4 Commit logical changes with `Refs #47`, push `npu/issue47-a`, open one PR to `dev`, and verify current-head CI without merging.
