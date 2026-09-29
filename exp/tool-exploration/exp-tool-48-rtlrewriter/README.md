# Sequential protocol scoreboard

Operation: sequential_scoreboard. RTLRewriter's correctness-before-cost principle
is extended to a cycle-exact FIFO contract. This bounded simulation does not
claim formal or general sequential equivalence.

The independent Python queue checks capacity, order, full replacement,
output stalls and reset flush. The source holds valid/data during backpressure.
Coverage counters must all be nonzero. Reset, capacity and ordering defects are
negative controls in tools/ic/tests/test_fifo_exploration.py.

Run the companion study in ../exp-tool-34-scalar/study.py. Both storage variants
are checked at 64/128-word capacities with 8,192 seeded cycles plus directed
fill/drain and simultaneous-transfer phases.
