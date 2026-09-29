# Tool 30: LFSR jump transformation

Hypothesis: GF(2) matrix exponentiation produces a parallel jump network with
less logic depth than chaining autonomous recurrence steps. Shared XOR-pair
extraction may reduce gates at a timing cost. Unshared matrix rows are an ablation
variant, not an additional counted tool. All starting states, including zero,
have exact defined behavior; no randomness or maximum-period claim is made.

Primary connection: Haramoto et al., *Efficient Jump Ahead for F2-Linear Random
Number Generators*, GERAD G-2006-62,
[author report](https://www.gerad.ca/en/papers/G-2006-62).
This implementation uses the report's binary-matrix baseline, not its faster
characteristic-polynomial sliding-window method. Hardware XOR sharing is a
separate bounded adaptation of DOI 10.1587/elex.11.20140934.

Predeclared cases: 32-bit state/poly 0x04C11DB7 advanced 64 steps and 64-bit
state/poly 0x1B advanced 128 steps. Compare unrolled to shared, area objective,
5 ns Basic flow, three paired runs. Each recurrence step shifts left and XORs
the polynomial when the old MSB is one. Matrix powers use repeated squaring.

Run `study.py --verify-only --container codex-sandbox-agent-workspace`; omit
`--verify-only` for physical studies. Exact affine proofs and independent bit-step
integer oracles cover the complete semantics. SAT cross-checks remain separately
visible, including timeouts. Keep failed and regressing cases in the final audit.
