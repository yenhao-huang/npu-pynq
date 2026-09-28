# Tool 32: canonical Mersenne reduction

Hypothesis: replace a generic unsigned remainder with balanced chunk additions,
end-around folding and a final conditional subtraction. The identity is
2**k congruent to 1 modulo 2**k-1. Intermediate bit widths follow conservative
integer bounds; a partial upper chunk is supported. Zero has a single encoding.

Primary connection: Campbell, Lin and Chen, *Low-cost hardware architectures for
mersenne modulo functional units*, ASP-DAC 2018, pp. 599-604,
[DOI 10.1109/ASPDAC.2018.8297388](https://doi.org/10.1109/ASPDAC.2018.8297388).
The [author institution record](https://experts.illinois.edu/en/publications/low-cost-hardware-architectures-for-mersenne-modulo-functional-un/)
describes residue reducers for checking arithmetic datapaths. This implementation
adapts the residue-reduction idea with ordinary binary adders; it does not claim
the paper's full-adder-only architecture or reproduce its ASIC results.

Contract: unsigned input width 8..128, exponent 2..16 strictly below width,
modulus 2**exponent-1, output 0..modulus-1. Native/folded variants are one operation.
Signed remainder and arbitrary divisors are outside this operation's contract.

Predeclared substantial cases: 16-bit mod-15 and 32-bit mod-255, area objective,
5 ns OOC clock, Basic optimization, three paired repeats, at most two recorded
implementation attempts. Registered boundaries match. A SAT timeout is unknown
and prevents physical acceptance. All failed cases remain in the audit.

```powershell
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-32-mersenne/study.py --verify-only --container codex-sandbox-agent-workspace
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-32-mersenne/study.py --container codex-sandbox-agent-workspace
```

Python integer-oracle tests cover 8/16/32/33/64/128-bit inputs, non-aligned chunks,
modulus boundaries, multiples, maximum values, exhaustive 8-bit inputs and seeded
random data. An incorrect zero canonicalization must be detected by the checker.
Actual study results belong in the report after execution; no PPA gain is assumed.

`config-prefix.json` retries the original unresolved case(s) with ordered
output-bit SAT. It uses already-proved lower-bit equalities and still requires
every bit, complete source guards and a successful bounded process. Use
`--configuration config-prefix.json` with this module's study command. Full
verification now passes; physical gain still requires all three measured pairs.
Historical failures and the original objective remain in the evidence.
