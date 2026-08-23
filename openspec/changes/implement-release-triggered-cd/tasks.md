## 1. Contract Tests

- [x] 1.1 Add failing host tests for standalone package allowlisting, required artifacts, and repository-independent layout; verify the focused package tests fail before implementation.
- [x] 1.2 Add failing host tests for normal, non-aligned, repeated board cases and deterministic evidence; verify the focused board-runner tests fail before implementation.
- [x] 1.3 Add a workflow contract test for `release.published`-only triggering, stable-release guards, trusted runner labels, artifact paths, protected environment, and Release assets; verify it fails against the current workflow.

## 2. Standalone Board Package

- [x] 2.1 Implement the allowlisted matrix-example package builder and verify focused package tests pass without writing tracked/generated files outside the selected output directory.
- [x] 2.2 Implement the non-interactive Phase 1C board runner with provenance verification, public runtime usage, required cases, JSON evidence, and failure exit codes; verify focused host tests pass with a fake physical runtime.
- [x] 2.3 Add a deterministic PowerShell deployment wrapper for a versioned PYNQ directory and verify syntax, dry-run validation, and absence of embedded credentials.
- [x] 2.4 Document standalone package contents and board execution in the example README and verify all referenced repository paths exist.

## 3. Release-Triggered CD

- [x] 3.1 Replace the tag-push build workflow with release-published validation, Vivado build, provenance verification, package upload, deterministic Release assets, protected board deployment, and evidence retention; verify the workflow contract test passes.
- [x] 3.2 Update CI/CD and file-tree rules for the release-only workflow and standalone example source layout; verify documentation agrees with workflow triggers and generated-artifact policy.

## 4. Validation

- [x] 4.1 Run focused example/package/workflow tests and the applicable repository Python suite, recording exact results.
- [x] 4.2 Run OpenSpec strict validation, `git diff --check`, ignored-artifact checks, and a secret/path scan; resolve every static failure.
- [x] 4.3 Record Vivado, timing, Release publication, SSH transfer, and physical-board validation as blocked until trusted runners and a new stable Release are available; provide the exact next commands/workflow gates.

## Validation Evidence

- PASS (2026-08-23): `python -m unittest discover -s src/test/tests -v` ran 58 tests.
- PASS (2026-08-23): `python -m unittest discover -s examples/matrix-multiplication/tests -v` ran 14 tests, including direct package CLI import/default inference, workflow, fake-runtime board cases, and PowerShell DryRun contracts.
- PASS (2026-08-23): PowerShell parser accepted `examples/matrix-multiplication/deploy_release.ps1`; the DryRun test proved no archive, evidence file, or network command was produced.
- PASS (2026-08-23): strict OpenSpec validation, YAML parsing, Python compilation, `git diff --check`, generated-artifact tracking check, and production-file secret/absolute-path scan.
- LOCAL LIMITATION: `make -C src/test model` could not start because GNU Make is unavailable on this Windows host; both Python commands invoked by that target passed directly.
- BLOCKED: Vivado synthesis, implementation, routed timing, DRC, BIT/HWH generation, and `build_evidence.txt` require the trusted `self-hosted, vivado` runner. The `build-overlay` job runs `vivado -mode batch -nojournal -nolog -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl` after Release validation.
- BLOCKED: GitHub Release asset publication requires this change on the default branch and publication of the next stable `vMAJOR.MINOR.PATCH` Release whose commit is contained in `origin/main`.
- BLOCKED: SSH transfer and physical PYNQ-Z1 execution require approval of the protected `pynq-z1-production` environment and an online `self-hosted, pynq-z1` runner with SSH configured. The `board-validation` job must upload `board-evidence.json`; a missing runner, missing evidence, or failed normal/non-aligned/repeated case keeps CD failed and leaves `current` unchanged.
