# Current expanded acceptance status

Snapshot: audit 7d5016. Whole-study acceptance remains incomplete.

- Implemented operation inventory: 50; semantic/exercise review remains.
- Unique declared comparisons: 59.
- Complete source-bound physical comparisons: 37.
- Qualifying families with both substantial configurations: seven.
- All-case objective geometric benefit: **undefined** because evidence is missing.
- Active work: popcount and CSD physical studies, dot-product proof retry,
  and a source-bound DSP timing-coverage diagnosis.

Coverage-bearing records now receive an automatic timing audit. Historical
records without coverage remain explicitly unaudited and need independent review.
The 32-bit matrix baseline is rejected for 566 total versus 558 clocked cells;
the eight differences are DSP48E1 cells whose active register use is unresolved.
No timing exemption is assumed and no failed comparison is removed.

## Missing complete comparisons

| Generator | Baseline geometry | Candidate | Objective | Recorded status |
| --- | --- | --- | --- | --- |
| synth_adder_tree | width=16, lanes=16 | balanced | throughput | no compatible complete study envelope |
| synth_adder_tree | width=16, lanes=16 | compressor | throughput | no compatible complete study envelope |
| synth_adder_tree | width=32, lanes=16 | balanced | throughput | no compatible complete study envelope |
| synth_adder_tree | width=32, lanes=16 | compressor | throughput | no compatible complete study envelope |
| synth_csd_multiplier | width=16, constant=255 | csd | area | verified |
| synth_csd_multiplier | width=32, constant=65535 | csd | area | correctness_not_established |
| synth_fir | width=16 | symmetric | area | correctness_not_established |
| synth_fir | width=32 | symmetric | area | correctness_not_established |
| synth_dot_product | width=16, lanes=8 | compressor | throughput | correctness_not_established |
| synth_dot_product | width=32, lanes=8 | compressor | throughput | correctness_not_established |
| synth_booth_multiplier | width=16 | radix4 | area | correctness_not_established |
| synth_booth_multiplier | width=32 | radix4 | area | correctness_not_established |
| synth_popcount | width=64 | tree | throughput | no compatible complete study envelope |
| synth_popcount | width=128 | tree | throughput | no compatible complete study envelope |
| synth_leading_zero | width=64 | tree | throughput | measurement_failed; verified |
| synth_leading_zero | width=64 | binary_search | throughput | measurement_failed |
| synth_leading_zero | width=128 | binary_search | throughput | measurement_failed |
| synth_divider | width=16 | serial | area | measurement_failed |
| synth_divider | width=16 | radix4 | throughput | measurement_failed |
| synth_constant_modulo | width=16, exponent=4 | folded | area | verified |
| synth_constant_modulo | width=32, exponent=8 | folded | area | correctness_not_established |
| synth_systolic_tile | width=32, size=2 | systolic | area | Timing coverage rejected: Some sequential cells are outside the measured clock; verified |

## Required independent review

- Substantive distinct operation purposes and direct PPA classification.
- Meaningful negative tests and actual per-operation experiment coverage.
- Historical timing coverage and report authenticity.
- Declaration timing, objective preservation and full historical inventory.
- Required documents, OpenSpec, PR state and usage stop rule.

The machine-readable snapshot is [acceptance-audit.json](evidence/acceptance-audit.json).
Rerun after new evidence arrives. The controlling requirements remain
[acceptance-50.md](acceptance-50.md).
