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

## Signed datapath and storage progress

- [x] Implement FIR-window symmetry, signed dot-product reduction and saturating ALU sharing.
- [x] Verify integer extrema, odd structures and signed/overflow mutation controls.
- [x] Complete distributed FIFO physical pairs at both sizes with protocol evidence.
- [ ] Complete FIR/dot proofs and ALU/network physical studies before claiming gains.
- [ ] Implement remaining fourteen distinct operations and finish global acceptance.

## Linear datapath progress

- [x] Implement parallel CRC and exact LFSR jump with shared-XOR alternatives.
- [x] Replace the duplicate compressor proposal with source-bound affine proof.
- [x] Run real affine positive/negative controls and replay the counterexample.
- [x] Retain completed saturation/barrel measurements that miss family gates.
- [ ] Complete CRC/LFSR/argmax physical studies and full all-case acceptance.

## Iterative arithmetic progress

- [x] Implement operations 22, 31 and 44 with independent cycle/result checking.
- [x] Verify 16/32-bit core and registered observations across three architectures.
- [x] Complete argmax physical gains at both sizes and all three pairs (family four).
- [x] Preserve completed CRC/LFSR below-threshold measurements and proof outcomes.
- [ ] Finish multiplier/divider physical studies and remaining eight operations.
- [ ] Establish two more qualifying families and every global acceptance gate.

## Signed multiplier progress

- [x] Implement signed Booth operation 21 with odd-width and extrema coverage.
- [x] Exercise independent integer oracles and two recoding mutation controls.
- [ ] Complete substantive SAT/physical studies; no multiplier family win yet.
- [ ] Implement remaining seven distinct operations and complete all gates.

## Control microarchitecture progress

- [x] Implement cyclic phase FSM operation 36 with binary and one-hot encodings.
- [x] Extend the existing scoreboard with independent phase/hold/wrap/reset checks.
- [x] Verify both 64/128-state cores and fixtures; preserve four mutation controls.
- [x] Preserve final Booth large-width SAT timeouts without claiming physical gains.
- [ ] Complete phase-controller physical repeats and remaining six operations.
- [ ] Finish the six-family improvement and whole-study aggregate gates.

## Timing and completed control evidence

- [x] Implement operation 43 with actual routed report and clock coverage checks.
- [x] Verify 16/32-bit real positive cases and an unclocked-domain negative case.
- [x] Complete phase-controller pairs at both sizes (fifth LUT-area qualifier).
- [x] Preserve FF increases and zero-LUT ratio bounds without fabricating values.
- [x] Retain completed divider failures and area/throughput tradeoffs.
- [ ] Complete remaining five operations and one more qualifying family.
- [ ] Complete all-case aggregation, ablations and final acceptance audit.

## Register-file progress

- [x] Implement operation 35 with explicit banking, reset and conflict contracts.
- [x] Verify independent memory semantics at both substantial configurations.
- [x] Detect arbitration, validity, row, read/write ordering and fixture mutations.
- [ ] Complete the register-file physical study without hiding any conflicts.
- [ ] Complete remaining four operations and all-case acceptance.


## Backpressure pipeline progress

- [x] Implement operation 37 with elastic and skid-stage architectures.
- [x] Verify both substantial configurations, all core/fixture roles and mutations.
- [x] Establish six qualifying families with all paired runs (including multiplier).
- [ ] Complete ongoing memory and skid physical studies.
- [ ] Complete remaining operations 38, 47 and 50 and the all-case acceptance gate.
