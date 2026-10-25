# PYNQ Development State

Run ID: issue119-onnx-mlir-20261025
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-10-25
Scope: Implement the approved ONNX-MLIR plan, starting with compatibility and import.
OpenSpec change: `incorporate-onnx-mlir`

Last updated: 2026-10-25

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | Approved plan: `docs/plans/onnx-mlir-incoporate/plan.md`; user requested implementation. | Existing #119 is open, assigned to yenhao-huang, claimed by agent a; branch `npu/issue119-a` in its dedicated worktree. No new branch or commit. |
| 1. Read Context and Rules | completed | Read skills, convention, filetree, state/environment, matrix, validation, and issue rules. | Export/compiler and eventual runtime/examples/board affected. Large pre-existing staged/unstaged changes are preserved. |
| 2. Prepare OpenSpec Change | completed | OpenSpec 1.15.0 strict validation passed; apply instructions report ready; all returned context files read. | Selected change: `incorporate-onnx-mlir`, 0/11 tasks complete. |
| 3. Implement | blocked | Tasks 1.1-1.3 completed: new `src/export/compilers/onnx_mlir/`, focused import tests, `docs/onnx-mlir/README.md`, and filetree documentation. | Task 1.4: pinned upstream image lacks ARM target; LLVM 24/22 interchange unverified. Production export, vendor, driver, RTL and ORT sources unchanged by this run. |
| 4. Validate | blocked | 18 focused/import/boundary tests passed with `NPU_TEST_ONNX_MLIR=1`; existing compiler/boundary tests: 19 run, 8 skipped (native LLVM absent). Real pinned SmolLM2 import succeeded, producing `build/onnx-mlir/smollm-import/model.onnx.mlir`; source hash matches pin. Strict OpenSpec validation passed. | ARM Sigmoid object generation fails: `No available targets are compatible with triple armv7-unknown-linux-gnueabihf`. Full numeric, partition and board gates outstanding. RTL lint/sim not applicable to this import-only change; no push requested. |
| 5. Handoff | completed | OpenSpec records 3/11 tasks complete, remaining gates explicit. Tool and command documentation in `docs/onnx-mlir/README.md`. User authorized a local commit containing only this change. | Next: provide an ARM-enabled compiler build and prove the unchanged NPU input boundary before production integration. No push/PR/merge. |

## Previous Run Record

Run ID: issue47-real-resnet18-20260903
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-09-03T23:42:00+08:00
Scope: Download a pinned public pretrained ResNet-18, convert it into the
repository QuantizedGraph/NPU package, validate real input execution, and make
the example package consume its gitignored model workspace for Issue #47.
OpenSpec change: `add-real-resnet18-import-export`

Last updated: 2026-09-04T00:31:50+08:00

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | Issue #47 is open, assigned to `yenhao-huang`, claimed by agent-id `a`, and formally blocks #7. | Branch `npu/issue47-a`; worktree `worktrees/npu-issue47-a`; base `origin/dev` at `5625a28`. |
| 1. Read Context and Rules | completed | Read `AGENTS.md`, `docs/rules/filetree.md`, pynq-dev skill and all required references, Issue #47, and Phase 2A source/OpenSpec inventory. | Affected areas: export/compiler, numeric conversion, examples/docs; runtime and board gates depend on observed behavior. |
| 2. Prepare OpenSpec Change | completed | OpenSpec CLI 1.12.0 reports 4/4 artifacts complete; strict validation passed for `add-real-resnet18-import-export`. | Two new capabilities freeze acquisition/conversion and example-workspace behavior. |
| 3. Implement | completed | OpenSpec tasks 1.1-4.4 complete. Official checkpoint: 46,830,571 bytes, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Two conversions produced byte-identical manifest, payload, input, and provenance. Full real model exported and reloaded; independent vectorized reference and `NPUModelRuntime` agreed exactly for `stem.relu`, `layer1.1.relu`, and `logits` over a `(1, 224, 224, 3)` signed-INT8 input. | Export certificate found and the converter corrected a near-dead stem channel that initially exceeded INT32 by 302,207; the safety certificate was preserved, not weakened. Model archives and acceptance JSON are independently byte-reproducible. |
| 4. Validate | completed with external gates recorded | Focused Issue #47 tests: 16 passed. Full suites: 107 `src/test`, 16 matrix example, 8 ResNet example tests passed (131 total). Seven Icarus RTL simulations passed. Strict OpenSpec validation and `git diff --check` passed. Generated files are ignored and no credential literal was found. | Local Verilator gate is blocked because the installed launcher lacks `verilator_bin`; GitHub CI must run the authoritative Linux lint. Physical command failed closed before execution because `build/vivado/npu_matrix/artifacts/npu_matrix.bit` (and matching HWH/manifest) is absent; no board evidence was written and no host result was relabeled. |
| 5. Handoff | completed | Commit `0a4fbfdf0845c941dd6bf7eb8877b146f31a8612` was pushed on `npu/issue47-a`; PR #48 targets `dev`, is mergeable, and its Linux `lint-and-simulate` check passed. | PR remains open and unmerged. Repository policy calls for squash merge after human review. |
