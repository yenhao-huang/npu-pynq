# Tool 20: signed dot-product accumulation

Hypothesis: full-width 3:2 carry-save stages shorten the accumulation dependency
chain after signed products, potentially improving throughput at a resource cost.
A balanced carry-propagate variant is available for ablation. Variants count as
one tool. Packed inputs contain a_i then b_i for each lane, in increasing words.
Products and the final sum retain full precision; sign extension precedes reduction.

Paper: Thomas B. Preusser, *Generic and Universal Parallel Matrix Summation with
a Flexible Compression Goal for Xilinx FPGAs*, FPL 2017,
[author manuscript](https://arxiv.org/abs/1806.08095), DOI 10.23919/FPL.2017.8056834.
The implementation adapts matrix-compression arithmetic with ordinary 3:2 cells.
It does not replicate the paper's device-specific counter selection heuristic.

Predeclared PPA: 8 pairs of signed 16/32-bit operands, serial baseline, compressor
candidate, logic-only multiplier policy, throughput objective, 5 ns Basic flow,
three physical pairs and two bounded attempts per run. Registered boundaries match.
Independent integer oracles cover signed extrema, non-power-of-two lane counts
and seeded data. Removing signed multiplication is a failing mutation control.

Run `study.py --verify-only --container codex-sandbox-agent-workspace`; omit
`--verify-only` for gated physical execution. SAT timeout remains unknown.
No improved II, latency, PPA or power is assumed from the architecture label.

`--configuration config-products.json` selects the separately recorded
300-second bitwise product-abstraction retry. Geometry, architectures, resource
policy and throughput objective are identical to the original declaration.
Only identical elaborated multiplication cells can be shared before abstraction.
Every output bit must pass; unresolved proofs still block physical execution.

`config-prefix.json` retries the original unresolved case(s) with ordered
output-bit SAT. It uses already-proved lower-bit equalities and still requires
every bit, complete source guards and a successful bounded process. Use
`--configuration config-prefix.json` with this module's study command. Full
verification now passes; physical gain still requires all three measured pairs.
Historical failures and the original objective remain in the evidence.

## Dot-product physical completion

Study 515cd1 completes all three paired runs for the original 16/32-bit, eight-
lane dot-product comparisons. The compressor candidate improves normalized
throughput by 9.30%/6.18% and grows LUTs by 7.56%/3.66%. Neither size reaches
the 15% gain threshold. All 12 source-bound timing audits pass. Failed physical
attempts remain in the study envelope, with the original throughput objectives.

Snapshot 2ab6a9 has 55/59 complete comparisons and nine qualifying families.
Only the two FIR and two Booth comparisons lack complete evidence; their formal
proofs remain unknown. Whole-case objective benefit is still undefined. There
are no active physical runs at this checkpoint. The historical SAT interface
audit now covers 98 source interfaces from 49 successful study proofs, all
matching their original source hashes. Hosted CI at 0342196 passes.
