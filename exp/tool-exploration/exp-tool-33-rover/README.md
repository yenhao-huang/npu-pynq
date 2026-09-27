# Tool 33: saturating add/subtract sharing

Hypothesis: one conditional-inversion adder can replace separately expressed
add/subtract paths while retaining exact signed overflow and clamping behavior.
This is an adaptation of arithmetic resource-sharing and equivalence-guided
rewriting from [ROVER](https://arxiv.org/html/2406.12421v1); it is not an e-graph
search engine or a reproduction of its optimization results.

Input packs two width-bit operands and a subtract flag. Output packs saturated
result and an overflow/underflow bit. Signed mode clamps to the two-complement
range; unsigned mode clamps to zero or the maximum word. Parameters are width
8..128, signedness and architecture. Shared and dual variants count as one tool.

Predeclared PPA: signed 32/64-bit add/subtract, area objective, matched 5 ns Basic
flow and registered boundaries, three pairs, at most two attempts per execution.
Independent integer tests also check unsigned arithmetic and 16-bit operands,
all pairs of extrema/zero/one, both operations and seeded random inputs.

Run `study.py --verify-only --container codex-sandbox-agent-workspace`, then omit
`--verify-only` for gated physical execution. Report overflow as well as numeric
outputs, preserve source hashes and retain SAT/physical failures. Width reduction
or clamping differences cannot be used as PPA improvements.
