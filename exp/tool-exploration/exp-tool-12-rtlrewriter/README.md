# Wide combinational validation

Operation: `vector_equivalence`. Following RTLRewriter's correctness-before-PPA
workflow, compare independently compiled packed x/y interfaces. This is
simulation, not the paper's formal equivalence checker.

The shared reduction study uses 256/512 input bits and 8,192 random vectors,
plus all-zero, all-one, alternating, walking-one/zero and carry patterns.
The seed is 928. Source hashes and stimulus hashes bind each verdict.
Unknown output bits, missing rows, compile warnings about interface mismatch,
process errors and source changes fail the verdict.

Run `study.py` in `../exp-tool-14-rover/`. Focused negative controls and
reproducibility checks live in `tools/ic/tests/test_wide_exploration.py`.

## Arithmetic mutation controls

Run `python exp/tool-exploration/exp-tool-14-rover/controls.py`. At both original
16/32-bit,16-lane sizes, the correct balanced/compressor cores pass directed and
seeded vector checks. Replacing the first addition with subtraction or shifting
the first compressor carry by two instead of one yields concrete mismatches.
All eight expected verdicts match. These controls check failure detection; they
are not additional physical measurements or formal proofs. See the goal evidence
file reduction-mutation-controls.json.
