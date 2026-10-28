# Tasks

## 1. Compatibility and source import

- [x] 1.1 Add deterministic source inventory and conversion diagnostics; verify fixture tests and inventory the pinned SmolLM2 model.
- [x] 1.2 Add pinned tool discovery and ONNX-to-MLIR import with retained diagnostics; verify failure cases and import a real fixture.
- [x] 1.3 Document the new compiler boundary and runnable inventory/import commands; verify paths and CLI smoke tests.
- [x] 1.4 Prove ARM32 compilation and an unchanged-NPU-compatible contraction; record compiler commands, IR, and execution evidence.

## 2. CPU conversions

- [x] 2.1 Define implementation sub-issues for operator conversion and integration under #119 before changing production model behavior; verify ownership and dependency links.
- [x] 2.2 Compile normalization, residual, sigmoid, and dynamic quantization with documented tolerances and reference edge-case tests.
- [x] 2.3 Compile rotary embedding and GQA with cache-update, position, and masking tests for prefill and decode.
- [x] 2.4 Adapt imported projections to existing NPU partition IR and verify instruction/weight parity without vendor edits.

## 3. Production integration

- [x] 3.1 Integrate the verified pipeline into SmolLM export without a manual bridge object; verify package and other-model regressions and update the WSL/macOS example guides and notebook.
- [x] 3.2 Validate prefill lengths 36/40/42 and multiple decode steps against reference logits and KV state on PYNQ-Z1; record memory and latency.
- [x] 3.3 Remove unused production ORT sources/assets after consumer audit and board parity; verify dependencies and preserve independent reference execution.

## Workflow follow-up

- Compatibility gate 1.4 passed: the pinned LLVM 24 importer emits generic ONNX IR that MLIR 22 parses. An imported Sigmoid executed as ARM32 hard-float code on PYNQ-Z1; an imported INT8 contraction passed unchanged NPU partition/lower with exact simulated parity. This does not validate the complete model on physical NPU hardware.
- Implementation sub-issues: #139 (CPU operators), #140 (package integration, blocked by #139), and #141 (board validation and cleanup, blocked by #140). All are assigned to yenhao-huang and claimed by agent a; the parent sub-issue API confirms all three relationships. Operator implementation uses worktree `npu-issue139-a`, branch `npu/issue139-a`, created from dev then fast-forwarded to the required #119 integration baseline.
- Operator evidence: `python3 -m unittest src.test.tests.test_onnx_lowering -v` in `npu-onnx-operators`: 8 passed, zero skipped, against ORT 1.20.1. Quantized values/scales/zero points and KV updates match exactly; floating tolerances are rtol <= 3e-6, atol <= 1e-6. Tests include zero/constant/tiny/large ranges, ties-to-even, rotary positions, prefill/decode causal masks and untouched cache layers.
- Full NPU evidence: `build/onnx-mlir/check_npu_parity.py` returned `FULL_NPU_PARITY_PASS`: all 480 tasks, instruction bytes, tiled weights, weight offsets and layouts match the legacy frontend exactly. Physical imported contraction returned `PHYSICAL_IMPORTED_CONTRACTION_PASS`, 1 job, 183 cycles, exact INT32 equality, zero error.
- End-to-end validation passed after preserving normalization division order, exact constant formatting, explicit FMA, eight-lane softmax reduction and grouped decode accumulation. Host and physical board both passed 12 cases: KV state and greedy tokens match exactly, max logit error 4.2e-5. Board peak RSS 381280 KiB; prefill 36/40/42 measured 30.879/17.094/17.719 seconds. Initial failed comparisons remain in ignored evidence logs.
- Final removal and documentation gates passed. Current source/workflow/executable experiment files have no references to the four deleted files or NativeOrtOperators. All 13 unused ORT-only experiment wrappers were removed; independent references remain. Source-only CLI and final physical runtime CLI passed. Notebook 32-token generation matched the independent full-source reference. See acceptance.md for the explicit CD reference review.
- Final regression suite ran 60 tests: 59 passed, one Docker-only import test skipped; the separate real-import run passed all 14 import tests. Documentation checks passed 23 links and 14 Bash blocks, example layout/AST/output checks and CD YAML parsing. OpenSpec strict validation and git diff --check passed.
- Keep issue #119 and the OpenSpec checklist synchronized with evidence.
- Do not push, open a PR, merge, or archive without the applicable authorization.
