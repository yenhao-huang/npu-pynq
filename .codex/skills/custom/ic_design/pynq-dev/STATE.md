# PYNQ Development State

Run ID: phase1a-systolic-20260822
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-08-22T01:15:00+08:00
Scope: Implement the signed saturating PE, parameterized systolic array, flow
control, and bit-accurate deterministic/randomized verification for Issue #3.
OpenSpec change: `implement-phase1a-systolic-array`

Last updated: 2026-08-22T09:53:47+08:00

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | Issue `#3` objective, scope, acceptance criteria, and dependency on Issue `#2` read from GitHub. | No AXI control, DMA, Vivado Tcl, bitstream, runtime, or board work; those belong to Issue `#4`. |
| 1. Read Context and Rules | completed | Required repository and `pynq-dev` context was read during the continued Phase 0/1 run. | Affected areas: numeric model consumer parity, RTL/interface, testbench, vectors, lint, and simulation. |
| 2. Prepare OpenSpec Change | completed | `implement-phase1a-systolic-array` has 4/4 artifacts complete and strict OpenSpec validation passed. | Issue `#3` claimed by GitHub user `yenhao-huang`, agent `a`; Phase 0 remains open in draft PR `#12`. |
| 3. Implement | completed | PE and array test-first failures recorded; signed saturating PE and parameterized rectangular array implemented. Seed `23063` generated 128 byte-reproducible golden cases with deterministic stalls. | OpenSpec implementation tasks 1.1 through 4.1 complete. |
| 4. Validate | blocked | `python -m pytest src/test/tests -q`: 39 passed. Direct Icarus `-g2012 -Wall` compile and simulation passed for PE, 2x2, 2x3, and randomized harness (`seed=23063`, `cases=128`). `npx.cmd -y @fission-ai/openspec@latest validate implement-phase1a-systolic-array --strict`: valid. `git diff --check`: pass; tracked generated-artifact, secret, and `docs/human/` scans: no matches. | Repository Make wrapper is blocked because GNU Make is not installed. The required Verilator lint gate is blocked because Verilator is not installed. These acceptance gates must run before Issue `#3` can be declared complete. Vivado, synthesis, bitstream, runtime, and board gates are not applicable to Phase 1A and belong to Issue `#4`. |
| 5. Handoff | completed | Implementation committed as `824e88f`; validation state committed as `7433d46`. Branch `npu/issue3-a` was pushed and draft PR `#13` opened to `dev`: https://github.com/yenhao-huang/npu_in_pynq/pull/13. Issue `#3` links the PR. | PR remains draft with no CI result yet. GNU Make and Verilator gates remain blocked; stacked dependencies are draft PRs `#12` and `#11`. Do not close Issue `#3`, delete its worktree/branch, or merge without the required evidence and authorization. |
