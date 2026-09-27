# Expanded implementation tasks

Previous 10-tool milestone: complete at 88acb1a, but insufficient for the current goal.
Current controlling contract: docs/goals/0928-tool-exploration/acceptance-50.md.

- [x] Reinspect clean worktree, active issue #80, PR #81 and current usage.
- [x] Define 50-tool scope, diverse benchmark and substantive physical-gain gates.
- [x] Upgrade physical measurement to clocked paths with complete FPGA resources.
- [x] Add large-interface simulation and SAT equivalence gates.
- [ ] Implement and validate 50 substantive tools (>=40 PPA related).
- [ ] Run >=12 microarchitecture families with >=2 substantive configurations.
- [ ] Demonstrate >=15% qualifying gains in >=6 families across both configurations.
- [ ] Complete three paired physical repeats and >=1.10 aggregate benefit ratio.
- [ ] Validate sequential protocol/reset/latency cases and negative controls.
- [ ] Publish complete results including regressions, failures and ablations.
- [ ] Update report, reproduction instructions, skill and PR; run final gates.

## Verified progress

32 exploration operations are implemented. Priority encoder and MCM have
complete source-bound correctness and three-pair physical gains at both
configurations. FIFO core/registered-fixture scoreboards pass, including
negative controls; its physical studies are ongoing. Netlist analysis has
actual generic/xc7 evidence at 16x64 and 32x128 storage sizes. The final 50-tool,
12-family, six-win and aggregate-benefit gates remain unchecked.

## Mersenne arithmetic checkpoint

- [x] Implement operation 32 with canonical residues and bounded chunk folding.
- [x] Verify independent integer oracles and a broken-zero mutation control.
- [x] Prove 16-bit mod-15 and preserve 32-bit mod-255 SAT timeout evidence.
- [ ] Complete the 32-bit proof and both physical configurations; no win claimed.
