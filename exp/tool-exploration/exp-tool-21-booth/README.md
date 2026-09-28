# synth_booth_multiplier

Primary paper: A. D. Booth, A Signed Binary Multiplication Technique,
Quarterly Journal of Mechanics and Applied Mathematics 4(2), 236-240 (1951),
DOI https://doi.org/10.1093/qjmam/4.2.236.
Original paper: https://www.ece.ucdavis.edu/~bbaas/281/papers/Booth.1951.pdf.

The generator implements adjacent-bit signed recoding and a paired-bit
adaptation with balanced partial-product sums. The paired version is not
attributed as a circuit reproduced from the 1951 paper. Native, radix2 and
radix4 are three variants of one operation. Operands are signed 8..64-bit
two's-complement words, including odd widths. Input packs a low / b high;
output is the exact full 2*width-bit product. Sign extension precedes negation
so the most-negative operand works. All variants request logic-only mapping.

Independent Python multiplication tests cover signed extrema, runs of ones,
odd sign groups and 2048 seeded random operand pairs per configuration. Mutated
sign extension and negative-two encoding must fail. Registered boundaries are
identical and use latency 1 / II 1 under the existing combinational fixture.

```sh
python exp/tool-exploration/exp-tool-21-booth/study.py --verify-only --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-21-booth/study.py --container codex-sandbox-agent-workspace
```

Predeclared cases: 16/32-bit native versus radix4, area objective, matched Basic
flow, three physical pairs. SAT and 8192 seeded random vectors gate physical
measurement. Timeout remains inconclusive; simulation alone cannot qualify a
combinational PPA win. No performance gain is assumed from fewer partial rows.

Use `study.py --configuration config-prefix.json --container
codex-sandbox-agent-workspace --verify-only` for a 300-second ordered-bit retry
per original configuration. This retry preserves the exact original cases,
architectures and area objectives. Prefix and AIG attempts are terminal without
a complete proof; partial obligations and timeouts cannot unlock physical
acceptance.

## Bounded AIG retry

Run `python exp/tool-exploration/exp-tool-21-booth/study.py --configuration
config-aig.json --verify-only --container codex-sandbox-agent-workspace` as one
command. Both original sizes and area objectives are retained with a 300-second
formal budget. Study 66caf5 times out on both cases; earlier prefix study 6d6ef5
proves only 14/32 and 14/64 obligations. These are unknown results.
