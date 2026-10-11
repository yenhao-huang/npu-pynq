# A/B problem dataset

Copied from the local RTLLM dataset at `~/.ab_private/RTLLM` used by
`deprecate/ab2/harness.py`. The original harness labels this checkout as RTLLM v2.0,
commit `9d5f61a`; the local copy has no Git metadata to independently verify
that revision. The upstream license is preserved in `LICENSE`.

Each category/problem directory contains `design_description.txt`,
`verified_*.v`, `testbench.v`, and any supporting stimulus files. These files
are input data. The Python loader lives separately in `src/problems/`.
Only the description is submitted to the model. References and testbenches
are used by the grader. Dataset corrections are applied to grading copies;
source data is not patched in place.

Use `--problems-dir PATH` to select another dataset with the same layout.
Problem directory names must be unique across categories.
