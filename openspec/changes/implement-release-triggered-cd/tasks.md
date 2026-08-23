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

## 5. PR CI Follow-up

- [x] 5.1 Record the failing PR and `dev` baseline lint evidence, classify every warning, and freeze the fix as behavior-preserving RTL cleanup.
- [x] 5.2 Eliminate controller width/latch warnings and narrowly annotate intentional AXI protection/reset warnings without global suppression or interface changes.
- [x] 5.3 Run all `npu_matrix` simulations and repository regressions, then require the current PR head's GitHub `lint-and-simulate` check to pass before merge.

### Follow-up Evidence

- FAILED (2026-08-23): PR runs `32617736644` and `32617836376` exit in `make -C src/test lint` with the same 16 Verilator warnings.
- BASELINE FAILED (2026-08-23): `dev` run `32617753539` reports the identical width, unused AXI protection, combinational latch, and mixed reset warnings.
- CONTRACT: No numeric, register-map, AXI, reset implementation, or board-visible behavior change is authorized; existing RTL simulations must remain exact.
- PASS (2026-08-23): all seven direct `iverilog -g2012 -Wall` simulations passed, including the AXI-Lite, controller, accelerator, PE, rectangular, deterministic, and 128-case random systolic tests.
- PASS (2026-08-23): 58 repository Python tests and 14 matrix-example/CD tests passed after the RTL cleanup.
- PASS (2026-08-23): current-head GitHub run `32619694272` completed `make -C src/test lint` with zero Verilator warnings and advanced to simulation.
- FAILED (2026-08-23): run `32619694272` exposed a testbench fixture path that assumed repo-root execution while the official Makefile runs from `src/test`; add a dual working-directory lookup and rerun CI.
- PASS (2026-08-23): current-head GitHub run `32619812743` completed `make -C src/test lint` and `make -C src/test sim`; PR #22 reports `CLEAN` and `MERGEABLE`.
