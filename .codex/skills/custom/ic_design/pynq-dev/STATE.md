# PYNQ Development State

Run ID: issue119-example-alignment-20261025
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-10-25
Scope: Align ResNet18 and matrix multiplication with create-example rules.
OpenSpec change: `align-resnet-matrix-examples`
Last updated: 2026-10-25

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | User requested both examples follow create-example. | Existing claimed #119 integration worktree; no commit/push requested. |
| 1. Read Context and Rules | completed | AGENTS, filetree, create-example, pynq-dev/dev and OpenSpec workflows read. | Preserve pre-existing staged and unstaged changes. |
| 2. Prepare OpenSpec Change | completed | OpenSpec 1.15.0 apply ready, five tasks. | Pure relocation: skip_specs. |
| 3. Implement | completed | Helpers relocated into export/runtime/test/CD; examples each retain two guides and one output-free notebook. CD, test discovery and active metadata callers updated. | Existing ignored model products and all pre-existing changes preserved. No hardware/vendor/numeric/human-doc edits. |
| 4. Validate | completed | 170 software tests: 150 passed, 20 existing environment skips; 48 focused tests passed. Isolated packages, dry run, pinned sources, boundary checks, 14 links, nine Bash blocks, notebook AST/outputs, CD YAML/PowerShell passed. Strict OpenSpec and diff whitespace passed. | Physical ResNet CLI and revised six-stage notebook: Samoyed, 35174 jobs. Matrix runtime: three cases passed, tile counts 1/4/4. Full matrix notebook/manifest CLI not rerun: original reused-overlay build manifest unavailable. No fabricated provenance. |
| 5. Handoff | completed | New board directory npu-example-alignment-20261025; 35 deployed source/notebook hashes checked. Full evidence and limits in openspec/changes/align-resnet-matrix-examples/acceptance.md. | No new Vivado/model build, macOS execution, release, commit, push, PR or archive. RTL gates not applicable: unchanged hardware and numerics. |

## Previous Run Record


Run ID: issue119-onnx-mlir-20261025
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-10-25
Scope: Complete the ONNX-MLIR migration, remove production ORT dependencies,
deploy the package, and verify physical PYNQ-Z1 execution.
OpenSpec change: `incorporate-onnx-mlir`

Last updated: 2026-10-25

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | Approved plan and user continuation through successful deployment/run. #119 assigned to yenhao-huang, claimed by agent a. | Integration worktree `npu-issue119-a`; sub-issues #139/#140/#141 have separate claimed branches/worktrees created from dev and advanced to the required #119 prerequisite. |
| 1. Read Context and Rules | completed | Repository and workspace AGENTS, applicable rules, pynq-dev/dev/OpenSpec/issue/deployment skills and references read. | Pre-existing staged/unstaged user changes preserved. No changes under docs/human, src/test/model, src/vendors or src/hw. |
| 2. Prepare OpenSpec Change | completed | OpenSpec 1.15.0 apply-ready; strict validation passed. All 11 tasks verified. | Do not archive, push, open a PR or merge without the applicable request. |
| 3. Implement | completed | Generic ONNX IR from pinned LLVM 24 importer parses with MLIR 22; production frontend consumes parsed IR before unchanged NPU partition. CPU operators, exact literals and fixed attention arithmetic compile to host/ARM. Four production ORT files and 13 unused ORT-only experiment wrappers removed after board validation. | Independent host ORT and NumPy/NPU references retained. WSL/macOS guides, output-free notebook, CD build and independently reviewed 32-token reference updated. |
| 4. Validate | completed | Final software suite: 60 run, 59 passed, one real-Docker test skipped there; separate NPU_TEST_ONNX_MLIR=1 import suite: 14/14 passed. Full 480-task NPU byte/weight/layout parity. Host and physical board: all 12 prefill/decode cases passed, exact KV/greedy IDs, max logits error 4.2e-5. Physical notebook: 32 tokens exact, 3840 calls, 1101600 jobs, 577335014 cycles, zero error. | Eleven isolated CPU tests include full-width prefill/decode tails. ARM hard-float ELF/symbol audit passed. Docs: 23 links and 14 Bash blocks passed; example layout/AST/output checks, YAML parse, source-only entry points, OpenSpec strict validation and diff whitespace passed. RTL/synthesis not applicable; unchanged hardware and no push. |
| 5. Handoff | completed | Deployed to `/home/xilinx/jupyter_notebooks/npu-onnx-mlir-20261025/`. Final 53 deployment hashes passed. After runtime cleanup, the public CLI ran physically and returned independent token 49 with 120 NPU calls and zero error. | ORT is absent from board PYNQ Python. Peak RSS 381280 KiB; observed prefills 30.879/17.094/17.719 s. Evidence: docs/plans/onnx-mlir-incoporate/acceptance.md and ignored build/onnx-mlir/*.json/logs. No commit, push, PR, merge, release or archive performed. |

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
