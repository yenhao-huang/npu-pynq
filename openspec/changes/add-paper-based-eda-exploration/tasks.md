# Expanded implementation tasks

Previous 10-tool milestone: complete at 88acb1a, but insufficient for the current goal.
Current controlling contract: docs/goals/0928-tool-exploration/acceptance-50.md.

- [x] Reinspect clean worktree, active issue #80, PR #81 and current usage.
- [x] Define 50-tool scope, diverse benchmark and substantive physical-gain gates.
- [x] Upgrade physical measurement to clocked paths with complete FPGA resources.
- [x] Add large-interface simulation and SAT equivalence gates.
- [x] Implement and validate 50 substantive tools (>=40 PPA related).
- [x] Run >=12 microarchitecture families with >=2 substantive configurations.
- [x] Demonstrate >=15% qualifying gains in >=6 families across both configurations.
- [ ] Complete three paired physical repeats and >=1.10 aggregate benefit ratio.
- [x] Validate sequential protocol/reset/latency cases and negative controls.
- [x] Publish complete results including regressions, failures and ablations.
- [x] Update report, reproduction instructions, skill and PR; run final gates.

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
- [x] Complete the 32-bit proof and both physical configurations; retain the measured result.

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
- [x] Finish multiplier/divider physical studies and remaining eight operations.
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
- [x] Complete phase-controller physical repeats and remaining six operations.
- [ ] Finish the six-family improvement and whole-study aggregate gates.

## Timing and completed control evidence

- [x] Implement operation 43 with actual routed report and clock coverage checks.
- [x] Verify 16/32-bit real positive cases and an unclocked-domain negative case.
- [x] Complete phase-controller pairs at both sizes (fifth LUT-area qualifier).
- [x] Preserve FF increases and zero-LUT ratio bounds without fabricating values.
- [x] Retain completed divider failures and area/throughput tradeoffs.
- [x] Complete remaining five operations and one more qualifying family.
- [ ] Complete all-case aggregation, ablations and final acceptance audit.

## Register-file progress

- [x] Implement operation 35 with explicit banking, reset and conflict contracts.
- [x] Verify independent memory semantics at both substantial configurations.
- [x] Detect arbitration, validity, row, read/write ordering and fixture mutations.
- [x] Complete the register-file physical study without hiding any conflicts.
- [ ] Complete remaining four operations and all-case acceptance.


## Backpressure pipeline progress

- [x] Implement operation 37 with elastic and skid-stage architectures.
- [x] Verify both substantial configurations, all core/fixture roles and mutations.
- [x] Establish six qualifying families with all paired runs (including multiplier).
- [x] Complete ongoing memory and skid physical studies.
- [ ] Complete remaining operations 38, 47 and 50 and the all-case acceptance gate.


## Component attribution and memory results

- [x] Implement operation 47 with complete factorial contrasts and failure handling.
- [x] Exercise source-bound storage effects at two substantial configurations.
- [x] Complete banked register-file physical repeats and twelve timing audits.
- [x] Establish a seventh qualifying family while preserving throughput losses.
- [x] Complete the additional capacity block and physical interaction analysis.
- [ ] Finish operations 38 and 50, global objective aggregation and full acceptance.


## Spatial matrix and completed buffering studies

- [x] Implement operation 38 as a true 2D signed matrix wavefront.
- [x] Verify 16/32-bit substantial core/fixture cases and nine mutation controls.
- [x] Complete skid physical repeats and preserve resource-growth disqualification.
- [x] Complete the FIFO capacity block and source-bound factorial interaction.
- [x] Audit all eighteen new skid/capacity physical timing records.
- [x] Complete matrix physical repeats and operation 50.
- [ ] Prove every whole-study acceptance gate, including the all-case aggregate.


## Complete operation inventory and acceptance gaps

- [x] Implement operation 50 with complete declaration and source-bound evidence audit.
- [x] Exercise the real 59-case inventory and retain 27 incomplete comparisons.
- [x] Reject invalid provenance, missing data and favorable-retry selection.
- [x] Reach 50 implemented operation names without claiming complete acceptance.
- [ ] Resolve remaining proof/measurement gaps and all-case aggregate benefit.
- [x] Review all operation semantics, experiment/negative-test coverage and provenance.
- [x] Complete final documents, OpenSpec validation where available and PR readiness.

## Proof completeness and physical coverage follow-up

- [x] Require all bitwise SAT obligations and preserve partial proof timeouts.
- [x] Validate conservative product abstraction against ten real SAT controls.
- [x] Reject unknown/partial operations and X/Z values before abstract acceptance.
- [x] Automatically audit physical records carrying timing-coverage reports.
- [x] Complete prefix, one-hot and matrix physical pairs; preserve regressions.
- [x] Keep the matrix32 DSP clock-count mismatch rejected pending diagnosis.
- [ ] Resolve remaining proof and physical gaps, including running retries.
- [ ] Establish all-case benefit and complete independent semantic/evidence review.

## Source semantics and inactive DSP storage

- [x] Require separate source-netlist validation before miter simplification.
- [x] Validate five normalization modes with 45 actual semantic controls.
- [x] Record arithmetic-normalized retries without changing declarations/objectives.
- [x] Audit complete per-cell DSP properties and actual active/inactive controls.
- [x] Complete CSD16, modulo16 and popcount64 physical pairs and timing audits.
- [x] Bound Windows run-publication retries without losing failed evidence.
- [ ] Resolve candidate matrix CREG usage and remaining large arithmetic proofs.
- [ ] Complete current popcount/adder physical retries and full all-case acceptance.

## Larger integration of the original operations

- [x] Exercise tools 01-10 on the original CSD16 study with actual measurements.
- [x] Exhaust all 65,536 inputs and reject a highest-output-bit mutation.
- [x] Record an explicit negative control for each original operation.
- [x] Bind the core hashes to the separate registered study's formal proof.
- [x] Preserve combinational scope and the original 59-case denominator.

## Bounded DSP C-register audit

- [x] Capture direct static controls and reject unknown or C-selected mux paths.
- [x] Verify unused C and active unclocked C through real routed controls.
- [x] Preserve all unconstrained-endpoint checks and rejected development data.
- [x] Complete the new matrix32 physical retry and all six associated audits.

## Ordered-bit proof retry

- [x] Add ordered-bit proof with complete obligation and source-safety checks.
- [x] Exercise ten real controls including first/last-bit mutations.
- [x] Prove original CSD32, modulo32 and both dot-product cases.
- [x] Complete all four original adder comparisons and matrix32 timing evidence.
- [ ] Complete the newly unlocked physical studies and whole-case acceptance.

- [x] Reject hidden/unobserved ports in both SAT source interfaces; run 48 actual
  interface controls and 60 normalization controls. Re-elaborate 88 historical
  source interfaces without altering their old proof records.

## Physical retry checkpoint

- [x] Complete original leading-zero64/tree and 64/128 binary-search comparisons.
- [x] Complete original divider16 folding/radix4 comparisons with all timing audits.
- [x] Retain FIR/Booth prefix, arithmetic and AIG unknowns without shrinking cases.
- [x] Extend historical SAT interface audit to 94 sources from 47 successful runs.
- [ ] Complete dot-product physical runs and establish FIR/Booth equivalence.
- [ ] Reach full-case objective benefit and finish independent operation review.

## Affine source guard review

- [x] Reject source partial semantics that disappear during affine synthesis.
- [x] Exercise eight source controls and replay the wrong-polynomial witness.
- [x] Reprove four distinct historical CRC/LFSR pairs with complete source guards.
- [ ] Finish the remaining substantive operation and whole-case acceptance review.

## Dot-product and reduction checkpoint

- [x] Complete both original dot-product physical comparisons and 12 timing audits.
- [x] Extend historical SAT interface checks to 98 source interfaces.
- [x] Detect balanced-adder and compressor-carry mutations at both original widths.
- [ ] Establish complete proofs and physical evidence for FIR and Booth.
- [x] Finish operation11 and16-50 semantic review.
- [ ] Establish full-case acceptance.

## Complete operation and provenance review

- [x] Document the purpose, contract and failure-path coverage of all 50 operations.
- [x] Add substantial positive and mutation controls for operations 11, 16-18 and 23-28.
- [x] Require complete-interface and total-binary evidence from accepted proofs.
- [x] Re-elaborate 98 historical SAT sources under the current proof contract.
- [x] Authenticate 182 preserved pre-coverage physical records against raw reports.
- [x] Verify 57 parent declarations precede 362 successful physical child runs.
- [x] Pass the complete 582-test Python suite and the required RTL lint/sim target.
- [ ] Prove and measure FIR16/32 and Booth16/32; whole-case benefit remains undefined.
