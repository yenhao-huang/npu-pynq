# Tasks

## 1. Compatibility and source import

- [x] 1.1 Add deterministic source inventory and conversion diagnostics; verify fixture tests and inventory the pinned SmolLM2 model.
- [x] 1.2 Add pinned tool discovery and ONNX-to-MLIR import with retained diagnostics; verify failure cases and import a real fixture.
- [x] 1.3 Document the new compiler boundary and runnable inventory/import commands; verify paths and CLI smoke tests.
- [ ] 1.4 Prove ARM32 compilation and an unchanged-NPU-compatible contraction; record compiler commands, IR, and execution evidence.

## 2. CPU conversions

- [ ] 2.1 Define implementation sub-issues for operator conversion and integration under #119 before changing production model behavior; verify ownership and dependency links.
- [ ] 2.2 Compile normalization, residual, sigmoid, and dynamic quantization with documented tolerances and reference edge-case tests.
- [ ] 2.3 Compile rotary embedding and GQA with cache-update, position, and masking tests for prefill and decode.
- [ ] 2.4 Adapt imported projections to existing NPU partition IR and verify instruction/weight parity without vendor edits.

## 3. Production integration

- [ ] 3.1 Integrate the verified pipeline into SmolLM export without a manual bridge object; verify package and other-model regressions and update all three example guides.
- [ ] 3.2 Validate prefill lengths 36/40/42 and multiple decode steps against reference logits and KV state on PYNQ-Z1; record memory and latency.
- [ ] 3.3 Remove unused production ORT sources/assets after consumer audit and board parity; verify dependencies and preserve independent reference execution.

## Workflow follow-up

- Compatibility gate 1.4: the pinned image imports the complete source but has no ARM target; direct ARM object generation fails with `No available targets are compatible with triple`. LLVM 24 to the existing LLVM 22 NPU boundary remains unverified. See `docs/onnx-mlir/README.md` for reproduction and observed versions.
- Keep issue #119 and the OpenSpec checklist synchronized with evidence.
- Do not push, open a PR, merge, or archive without the applicable authorization.
