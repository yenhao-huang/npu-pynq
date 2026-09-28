# Current expanded acceptance status

Snapshot: 2ab6a9. Whole-study acceptance remains incomplete.

- Implemented operation inventory: 50; semantic/exercise review remains.
- Declared comparisons: 59; complete physical comparisons: 55.
- Qualifying families at both substantial configurations: 9.
- All-case objective geometric benefit: undefined (incomplete evidence).

The original ten operations now have complete larger CSD16 integration evidence.
All four original adder comparisons and matrix32 have complete timing audits;
their resource/throughput regressions remain visible. CSD32 and modulo32 now complete both qualifying families. Leading-zero and
divider16 retries complete five more comparisons; their regressions remain.
Both dot-product sizes now have complete physical pairs and timing audits, but
throughput gains stay below the qualifying threshold. FIR/Booth remain unproved after bounded
normalization attempts. Complete-interface controls reject hidden-port proofs.
Historical missing-coverage records and failed retries remain in the inventory.

## Missing complete comparisons

| Generator | Geometry | Candidate | Objective | Recorded status |
| --- | --- | --- | --- | --- |
| synth_fir | width=16 | symmetric | area | correctness_not_established |
| synth_fir | width=32 | symmetric | area | correctness_not_established |
| synth_booth_multiplier | width=16 | radix4 | area | correctness_not_established |
| synth_booth_multiplier | width=32 | radix4 | area | correctness_not_established |

## Remaining independent review

- Distinct substantive operation purposes and direct PPA classification.
- Meaningful negative tests and actual per-operation experiment coverage.
- Historical timing coverage, report authenticity and declaration history.
- Complete documents, OpenSpec validation and PR readiness.

See [acceptance-audit.json](evidence/acceptance-audit.json) and the controlling
[acceptance-50.md](acceptance-50.md). No failed or incomplete case is removed.
