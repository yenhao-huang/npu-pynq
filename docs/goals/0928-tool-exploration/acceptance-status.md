# Current expanded acceptance status

Snapshot: 11c679. Whole-study acceptance remains incomplete.

- Implemented operation inventory: 50; semantic/exercise review remains.
- Declared comparisons: 59; complete physical comparisons: 41.
- Qualifying families at both substantial configurations: seven.
- All-case objective geometric benefit: **undefined**.
- Active physical retry: all four original adder-tree cases.

Matrix32 retry 514661 preserves identical resources and throughput regressions.
The baseline's eight unused ADREG/DREG defaults now pass a narrow DSP audit.
The candidate still has four additional DSPs with CREG=1 and USE_MULT=MULTIPLY;
their C-register use is unresolved. The comparison remains incomplete. Old
reports remain unchanged. Historical coverage omissions require review.

## Missing complete comparisons

| Generator | Geometry | Candidate | Objective | Recorded status |
| --- | --- | --- | --- | --- |
| synth_adder_tree | width=16, lanes=16 | balanced | throughput | no compatible complete study envelope |
| synth_adder_tree | width=16, lanes=16 | compressor | throughput | no compatible complete study envelope |
| synth_adder_tree | width=32, lanes=16 | balanced | throughput | no compatible complete study envelope |
| synth_adder_tree | width=32, lanes=16 | compressor | throughput | no compatible complete study envelope |
| synth_csd_multiplier | width=32, constant=65535 | csd | area | correctness_not_established |
| synth_fir | width=16 | symmetric | area | correctness_not_established |
| synth_fir | width=32 | symmetric | area | correctness_not_established |
| synth_dot_product | width=16, lanes=8 | compressor | throughput | correctness_not_established |
| synth_dot_product | width=32, lanes=8 | compressor | throughput | correctness_not_established |
| synth_booth_multiplier | width=16 | radix4 | area | correctness_not_established |
| synth_booth_multiplier | width=32 | radix4 | area | correctness_not_established |
| synth_leading_zero | width=64 | tree | throughput | measurement_failed; verified |
| synth_leading_zero | width=64 | binary_search | throughput | measurement_failed |
| synth_leading_zero | width=128 | binary_search | throughput | measurement_failed |
| synth_divider | width=16 | serial | area | measurement_failed |
| synth_divider | width=16 | radix4 | throughput | measurement_failed |
| synth_constant_modulo | width=32, exponent=8 | folded | area | correctness_not_established |
| synth_systolic_tile | width=32, size=2 | systolic | area | Timing coverage rejected: Some sequential cells are outside the measured clock; verified |

## Remaining independent review

- Distinct substantive operation purposes and direct PPA classification.
- Actual per-operation experiments and meaningful negative-test coverage.
- Historical timing coverage, report authenticity and declaration history.
- Complete documents, OpenSpec validation and PR readiness.

See [acceptance-audit.json](evidence/acceptance-audit.json) and the controlling
[acceptance-50.md](acceptance-50.md). No failed or incomplete case is removed.
