# PYNQ Development State

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
