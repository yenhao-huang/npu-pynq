# Current expanded acceptance status

Snapshot: 049aa8. Whole-study acceptance remains incomplete.

- Implemented operation inventory: 50; semantic/exercise review remains.
- Declared comparisons: 59; complete physical comparisons: 46.
- Qualifying families at both substantial configurations: 7.
- All-case objective geometric benefit: None.

The original ten operations now have complete larger CSD16 integration evidence.
All four original adder comparisons and matrix32 have complete timing audits;
their resource/throughput regressions remain visible. Ordered-bit proofs unlock
CSD32, modulo32 and both dot-product sizes; physical retries are in progress.
Historical missing-coverage records and failed retries remain in the inventory.

## Missing complete comparisons

| Generator | Geometry | Candidate | Objective | Recorded status |
| --- | --- | --- | --- | --- |
| synth_csd_multiplier | width=32, constant=65535 | csd | area | correctness_not_established |
| synth_fir | width=16 | symmetric | area | correctness_not_established |
| synth_fir | width=32 | symmetric | area | correctness_not_established |
| synth_dot_product | width=16, lanes=8 | compressor | throughput | correctness_not_established; verified |
| synth_dot_product | width=32, lanes=8 | compressor | throughput | correctness_not_established; verified |
| synth_booth_multiplier | width=16 | radix4 | area | correctness_not_established |
| synth_booth_multiplier | width=32 | radix4 | area | correctness_not_established |
| synth_leading_zero | width=64 | tree | throughput | measurement_failed; verified |
| synth_leading_zero | width=64 | binary_search | throughput | measurement_failed |
| synth_leading_zero | width=128 | binary_search | throughput | measurement_failed |
| synth_divider | width=16 | serial | area | measurement_failed |
| synth_divider | width=16 | radix4 | throughput | measurement_failed |
| synth_constant_modulo | width=32, exponent=8 | folded | area | correctness_not_established; verified |

## Remaining independent review

- Distinct substantive operation purposes and direct PPA classification.
- Meaningful negative tests and actual per-operation experiment coverage.
- Historical timing coverage, report authenticity and declaration history.
- Complete documents, OpenSpec validation and PR readiness.

See [acceptance-audit.json](evidence/acceptance-audit.json) and the controlling
[acceptance-50.md](acceptance-50.md). No failed or incomplete case is removed.
