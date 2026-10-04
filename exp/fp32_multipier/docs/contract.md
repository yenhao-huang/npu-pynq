# FP12 experiment contract

- Requested path: `exp/fp32_multipier/` (directory spelling preserved).
- Format: E5M6, bias 15; custom FP12, not an IEEE standard interchange format.
- Combinational inputs a/b and output y, each 12 bits.
- Gradual underflow, round to nearest ties to even, signed zero, infinity on overflow.
- All NaNs and infinity times zero yield canonical positive quiet NaN 0x7e0; no payload propagation or exception flags.
- Positive finite values are exactly significand * 2^(effective_exponent-21).
- Baseline and candidates must match the same contract.
- Area-delay ranking: ADP = mapped cell area (um^2) * maximum input-to-output delay (ps); lower is better. Also retain the area-delay Pareto frontier.
- ASAP7 7.5T RVT TT NLDM, combinational AO/INVBUF/OA/SIMPLE libraries. Four original library files and SHA-256 hashes are recorded in build/pdk/sources.json.
- Mapping target sweep: 900, 1200, 1700 ps. STA uses zero input/output delay, 10 ps input transition, 3.898 fF output load; no wire parasitics. These are declared experiment constraints, not claimed to exactly reproduce undocumented paper constraints.
- Power (if reported) is a vectorless estimate, not measured application power. It is not part of the ranking.
- Paper reference: bundled arXiv:2606.06530v2, FP16 E5M10, ASAP7 post-mapping. Baseline 121 um^2 / 1618 ps; final best area 79 um^2 and best delay 891 ps occur at DIFFERENT Pareto points. Absolute FP12 comparisons do not establish FP16 reproduction.
- Acceptance: more than 10 complete, functionally passing measured evaluations; report.md; reproduce.md for three best ADP designs. Track separately whether paper-like absolute PPA and relative improvement targets were reached.

## Validation

Independent Python nearest-grid search validated the integer oracle on 2,018,272 finite/special/sign checks. C++ oracle uses exact integer arithmetic and binade threshold comparisons. Verilator exhaustive runs cover all 16,777,216 input pairs for RTL and mapped cell models. Screening runs use 445,760 directed/random checks; final top three require exhaustive RTL and gate checks.

## First correction

Initial baseline failed a=0x001,b=0x001: a shifted-out half threshold became zero for shifts over 14, causing an erroneous increment. Explicitly zero results for shifts >14. Corrected baseline passes the full input domain in both RTL and mapped-netlist simulation.
