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
