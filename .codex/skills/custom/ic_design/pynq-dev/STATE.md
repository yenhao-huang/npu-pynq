# PYNQ Development State

Run ID: issue21-cd-lint-followup-20260823
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-08-23T12:30:00+08:00
Scope: Resolve the existing Verilator lint failures blocking Issue #21 PR #22
without changing numeric, register-map, AXI, reset, or board-visible behavior.
OpenSpec change: `implement-release-triggered-cd`

Last updated: 2026-08-23T13:12:00+08:00

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | Issue #21, branch `npu/issue21-a`, worktree, OpenSpec change, and PR #22 are reused. PR runs `32617736644` and `32617836376` and `dev` run `32617753539` fail with identical 16-warning Verilator output. | CI follow-up only; do not create a second issue branch or change hardware contracts. |
| 1. Read Context and Rules | completed | AGENTS.md, repository rules, pynq-dev context, development matrix, validation gates, issue workflow, OpenSpec context, RTL, and testbenches read. | Affected areas: RTL/interface and tests. Required gates: lint, all simulations, Python regressions, strict OpenSpec, and current-head GitHub CI. |
| 2. Prepare OpenSpec Change | completed | Existing `implement-release-triggered-cd` proposal, design, and tasks updated with behavior-preserving lint follow-up and objective failing-run evidence. | Explicit widths and complete combinational defaults are fixes; annotations are permitted only for intentional AXI protection/reset observations. |
| 3. Implement | completed | `npu_matrix_controller.sv` uses explicit zero-extension and a complete reduction-index default; AXI protection and mixed reset observations have narrowly scoped named Verilator annotations. The random testbench checks both official Makefile and repo-root fixture paths. | No global warning suppression, reset implementation, interface, register-map, or numeric change. |
| 4. Validate | in_progress | All seven direct `iverilog -g2012 -Wall` simulations PASS; 58 core Python and 14 example/CD tests PASS. GitHub run `32619694272` proves `make -C src/test lint` passes with zero warnings, then exposes only the fixture working-directory mismatch in simulation. | Rerun official GitHub simulation after the fixture-path fix. |
| 5. Handoff | pending | PR #22 remains open and unmerged. | Merge only after current-head `lint-and-simulate` passes. |
