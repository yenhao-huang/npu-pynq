# Expanded acceptance: fifty tools and diverse microarchitectures

This is the controlling acceptance contract for the expanded user goal. The
10-tool milestone in report.md is historical evidence, not completion of this goal.

## Mandatory gates

1. Implement and exercise 50 distinct registry operations. At least 40 must
   directly support PPA optimization. Parameter aliases and repeated scalar
   arithmetic helpers do not count as distinct tools.
2. Each tool needs a primary-paper connection, a bounded contract, useful negative
   tests and its own modular experiment under exp/tool-exploration/exp-tool-ID-PAPER/.
3. Benchmark at least 12 microarchitecture families with at least two substantive
   configurations each. Include arithmetic, reductions, selection/control,
   streaming/sequential and memory structures. Use 16/32-bit or wider arithmetic,
   or appropriately sized structures (e.g. 64/128 entries); the old 4-bit mux-adder
   is a regression test and cannot satisfy this diversity gate.
4. At least six families must show >=15% reduction in LUT area or >=15% increase
   in normalized throughput on BOTH configurations, in three paired physical
   runs. Area wins must not increase DSP/BRAM use or reduce throughput by >5%.
   Throughput wins must report latency and all resources; area-resource increases
   above 25% are trade-offs rather than qualifying unconditional wins.
5. Predeclare the configurations and objective per family before looking at PPA.
   Across all declared cases, the geometric mean of the selected objective's
   candidate/baseline benefit ratio must reach 1.10. Report every failure,
   timeout, regression and rejected candidate. Do not hide failed families by
   removing them from the denominator after seeing results.
6. Prove combinational equivalence with SAT where supported, and also use directed
   and seeded random vectors. Sequential changes need explicit reset, cycle,
   valid/ready, latency and initiation-interval scoreboards and negative controls.
   Random simulation is not a formal proof. No PPA win counts without matching
   source-bound correctness evidence.
7. Physical comparison must use the same target, tool build, flow, clock and
   directives. Use registered boundaries / OOC clocked paths instead of letting
   unconstrained board I/O dominate datapath timing. Normalize throughput by II.
   Repeat runs establish reproducibility, not independent statistical samples.
8. Update the survey, report.md, reproduce.md, exploration skill, OpenSpec and
   PR #81. Mark the PR draft while expanded acceptance is incomplete. Do not merge.
9. Keep the existing below-30%-remaining-usage stop rule. Save exact checkpoints
   and leave the goal incomplete when interrupted or when a gate is not proved.

## Counting and provenance

Each counted operation must have an independent engineering purpose and a
meaningful implementation, not merely forward to a renamed existing operation.
Architecture generators must emit real, parameterized RTL with documented
semantics; fixed templates without verified optimization behavior do not count as
finished experiments. Paper-inspired algorithms are identified as adaptations.
Fifty names alone never satisfy this contract.

## Planned operation inventory (not completion evidence)

IDs 01-10 are the existing operations; they must be integrated into larger studies.

| ID | Operation | Purpose |
| --- | --- | --- |
| 11 | clocked_ppa | Resource and register-to-register routed timing with II/latency |
| 12 | vector_equivalence | Large-interface directed and seeded random checking |
| 13 | yosys_equivalence | SAT proof of binary combinational equivalence |
| 14 | synth_adder_tree | Width-aware balanced multioperand addition |
| 15 | synth_compressor_tree | Carry-save reduction before a final carry propagation |
| 16 | synth_prefix_adder | Explicit prefix network exploration |
| 17 | synth_csd_multiplier | Signed-digit constant multiplication |
| 18 | synth_mcm | Shared adder graphs for multiple constants |
| 19 | synth_fir | FIR arithmetic and symmetry exploration |
| 20 | synth_dot_product | Product/reduction datapath generation |
| 21 | synth_booth_multiplier | Booth-recoded signed multiplication |
| 22 | synth_serial_multiplier | Area/II trade-off for iterative multiplication |
| 23 | synth_popcount | Hierarchical population-count network |
| 24 | synth_priority_encoder | Hierarchical priority selection |
| 25 | synth_leading_zero | Hierarchical leading-zero count |
| 26 | synth_barrel_shifter | Staged shift network exploration |
| 27 | synth_onehot_mux | Qualified one-hot selection architecture |
| 28 | synth_argmax_tree | Balanced compare/select tournament |
| 29 | synth_crc_parallel | GF(2) parallel CRC transform |
| 30 | synth_lfsr_jump | Algebraic jump-ahead recurrence |
| 31 | synth_divider | Iterative quotient/remainder architecture |
| 32 | synth_constant_modulo | Constant-modulus reduction |
| 33 | synth_saturating_alu | Exact overflow detection and saturation |
| 34 | synth_fifo | Storage inference and pointer architecture |
| 35 | synth_banked_regfile | Banking and port-conflict trade-offs |
| 36 | synth_fsm | Encodings with equivalent transition/output behavior |
| 37 | synth_skid_buffer | Ready/valid buffering without data loss |
| 38 | synth_systolic_tile | Pipelined spatial dot-product dataflow |
| 39 | netlist_profile | Mapped operator/resource attribution |
| 40 | critical_cone | Timing-cone extraction for targeted rewriting |
| 41 | fanout_analysis | High-fanout and replication opportunities |
| 42 | memory_inference | Detect register/LUTRAM/BRAM mapping consequences |
| 43 | timing_constraint_audit | Reject missing or misleading timing coverage |
| 44 | latency_throughput | Throughput/latency normalization and comparability |
| 45 | resource_tradeoff | Multi-resource FPGA dominance and budget enforcement |
| 46 | paired_repeat_summary | Reproducible paired-run aggregation |
| 47 | candidate_ablation | Attribute gains to individual transformations |
| 48 | sequential_scoreboard | Reset/protocol/latency-aware sequential checking |
| 49 | architecture_sweep | Checkpointed multi-configuration experiment execution |
| 50 | acceptance_audit | Evidence-backed gate evaluation across the whole study |

A planned entry can be replaced when the survey justifies a better operation,
but the reason must be recorded and the count/diversity/gain gates stay intact.
