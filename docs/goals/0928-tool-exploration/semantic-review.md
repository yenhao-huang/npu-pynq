# Operation and negative-control review

This review checks engineering purpose, implementation behavior and exercised
failure paths. Registry names, schemas and passing inventory checks alone do
not establish these properties. The review is incomplete: this checkpoint covers
operations 01-10 and 12-15. Operation 11, operations 16-50 and historical
physical coverage remain open.

## Original ten operations

The [larger workflow](evidence/wide-workflow.json) exercises every operation on
the same CSD16 cores used by the registered physical study. The exhaustive check
covers all 65,536 binary inputs. Its combinational PPA records support integration
testing and stay outside the registered comparison aggregate.

| ID | Distinct engineering purpose and bound | Exercised negative control |
| --- | --- | --- |
| 01 | Retrieve attributed rewrite advice and prerequisites; it does not rewrite or prove RTL. | Unknown topic returns no invented rule. |
| 02 | Compute exact add/subtract/multiply interval bounds for independent mathematical integers; caller must establish ranges and preserve RTL signedness. | 23 output bits cannot contain the CSD16 product; overflow is reported. Unit cases also cover negative ranges and zero. |
| 03 | Enumerate the caller-declared binary combinational interface, up to 16 input bits; require complete ordered transcripts and defined outputs. | Highest output-bit mutation fails at input zero; unknown outputs and incomplete execution are rejected by tests. |
| 04 | Run Vivado and collect routed combinational LUT/FF counts and input-to-output delay. This is a different scope from registered timing. | Sequential input is rejected; a timed-out process cannot produce an accepted record. |
| 05 | Compare declared measurement context; it does not authenticate external reports. | Mismatched part is incompatible; version mismatch is also tested. |
| 06 | Compute signed resource/delay deltas and percentage reductions after context/unit checks; a zero baseline has no percentage. | Wrong units are rejected; zero-baseline tests require a null ratio. |
| 07 | Compute a Pareto frontier with strict dominance, preserving equal points and trade-offs. | Duplicate candidate identities are rejected; a slower/smaller trade-off is not discarded by scalar ranking. |
| 08 | Form a weighted improvement score gated by correctness and hard limits. This is an adaptation, not a reproduced training loss. | Failed correctness yields ineligible/null reward; missing metrics and hard-limit violations are tested. |
| 09 | Choose feasible actions with cost-normalized UCB and deterministic unseen-action/tie handling. Estimated cost does not enforce execution time. | Insufficient budget selects nothing. Three finite-input arithmetic-overflow cases now reject selection. |
| 10 | Compose checking, measurement and comparison with source-hash binding and early failure. This is more than a renamed single operation. | Wrong RTL produces only the failed check child; source changes between checking and measurement are rejected in tests. |

Implementation inspected: the lint advice backend, exhaustive checker,
combinational measurement backend, PPA analytics backend and RTLRewriter pipeline.
Relevant regression: `tools/ic/tests/test_exploration.py`, 28 passing tests after
the selector correction. Paper connections and adaptation limits remain in the
per-operation READMEs, [survey](survey.md) and operation inventory.

## Selector issue found during review

Development run `68600c` returned an infinite UCB score from finite reward/cost
inputs and selected that action. This result is rejected evidence. The backend
now checks the calculated scores and handles integer-to-float overflow before
choosing an action. It does not clamp or silently rank an invalid score.

[Actual controls](evidence/selector-finite-controls.json) cover division overflow,
addition overflow and oversized visit-count conversion. The same runner also
rechecks a normal selection using the committed CSD study's measured reward and
elapsed cost. Run:

```sh
python exp/tool-exploration/exp-tool-09-rtlrewriter/controls.py
```

These results close this specific numerical issue. They do not complete the
remaining operation review, whole-case benefit or physical-evidence gates.

## Complete SAT interface review (operation 13)

Development run `0ccec1` incorrectly accepted a candidate whose extra output `z`
differed from the reference: the miter observed only `x` and `y`. That run is
explicitly rejected in interface-development-failure.json. The backend now
requires both pre-optimization source netlists to contain exactly input `x` and
output `y`, at the declared widths. A successful SAT result cannot override this
check. Extra unused inputs, extra outputs, inout ports and width mismatches fail.

The Docker Yosys 0.23 controls cover all six normalization modes: 48 interface
cases and 60 arithmetic/undefined-value cases match their expected verdicts.
The historical interface audit re-elaborated 88 source files from 44 successful
SAT study proofs, after matching each file set to its recorded source hash;
all have complete declared interfaces. This is an interface-only audit, not a
new SAT proof or retrospective change to a historical verdict. Its scope is the
evidence directory at execution; later studies require another audit.

Reproduce with `exp/tool-exploration/exp-tool-13-rover/interfaces.py`,
`bitwise.py` and `audit_interfaces.py`, each with
`--container codex-sandbox-agent-workspace`. See interface-controls.json,
normalization-interface-controls.json and historical-interface-audit.json in
`evidence/`. Operation 13's identified interface issue is fixed; review of the
remaining operations and historical physical coverage is still incomplete.

## Operation 15: exact affine proof and source semantics

Development run 14c87d incorrectly accepted `x[x] ^ x[x]` on an 8-bit input:
out-of-range indexing was simplified out before coefficient propagation. This
result is rejected, not accepted equivalence evidence. The affine checker now
composes `yosys_equivalence` self-checks for each source with its coefficient
algorithm. Each self-check requires complete interfaces, total binary source
semantics and matching source hashes before netlist optimization. A self-check
never replaces the cross-design affine proof. Runtime limits apply to each child
proof or synthesis process; a failed or unknown guard returns no affine proof.

Eight actual source controls match their expected verdicts, including cancelled
out-of-range indexing in either source separately. Valid constant division by
one passes. The original CRC64 pair still proves, a different polynomial fails,
and its concrete counterexample replays in Icarus. Nonlinear AND still rejects.
All four distinct historical CRC/LFSR source pairs (eight recorded study proofs)
prove again with the new guards, after validating original hashes. Old studies
and physical measurements remain unchanged; these are additional proof records.

```sh
python exp/tool-exploration/exp-tool-15-gf2/source_controls.py --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-15-gf2/recheck_studies.py --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-15-gf2/study.py
```

See `affine-source-controls.json`, `affine-source-rechecks.json`,
`affine-guarded-controls.json` and `affine-source-development-failure.json` under
`docs/goals/0928-tool-exploration/evidence/`. No new operation is counted.

## Operations 12 and 14: wide vectors and modular reductions

| ID | Distinct purpose and contract | Exercised failure path |
| --- | --- | --- |
| 12 | Simulate directed all-zero/one, walking-bit/carry patterns plus seeded random vectors on 1..4096-bit packed interfaces. Require a caller-confirmed complete combinational contract, all ordered transcript rows, defined outputs and unchanged sources. This is sampling, not proof or automatic verification of the caller's contract. | Subtraction replacing addition, unknown outputs and truncated input widths reject; process timeout/incomplete transcripts reject. New 16/32-bit reduction mutations give concrete input/output mismatches. |
| 14 | Generate unsigned sums modulo 2**width, with width 8..128 and lanes 3..128. Serial, balanced and carry-save implementations are architectures of one operation. Width truncation at each node preserves the modular sum; leftover nodes are forwarded at each level. The same registered wrapper supplies latency 1/II 1. | Replace the first balanced addition by subtraction, or shift the first compressor carry by two instead of one. Both fail at both original widths with concrete vector mismatches. Odd lane counts 3/7 and declared large 16-lane cases also pass existing integration tests. |

`exp/tool-exploration/exp-tool-14-rover/controls.py` runs eight actual checks at
16 and 32 bits with 16 lanes. All four correct candidates pass and all four
mutations fail with observed data mismatches (not compilation errors/timeouts).
See reduction-mutation-controls.json. The four existing physical comparisons
remain complete but do not qualify the family: balanced gains are small and
compressor resource growth exceeds the throughput gate. This review adds no
PPA win, no new operation and no sampled-equivalence claim of formal proof.
