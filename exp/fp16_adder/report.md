# FP16 adder experiment report

Target: RTLScout Section 7.1, binary16 addition (E5M10), local module `fpadd_fp16`.

Completed passing evaluations: 28; distinct passing RTL snapshots: 20.

| Run | Status | Area (um^2) | Delay (ps) | ADP | Estimated power (W) |
| --- | --- | ---: | ---: | ---: | ---: |
| 000_exact_baseline | pass | 125.44632 | 2110.819092 | 264794.4872771415 | 0.0009018913843 |
| 001_aligned_priority | failed | - | - | - | - |
| 002_aligned_priority | pass | 52.9254 | 1460.145508 | 77278.7850691032 | 0.0002251083642 |
| 003_aligned_case | pass | 56.45376 | 1311.40979 | 74034.0135463104 | 0.0002913230855 |
| 004_shared_priority | pass | 51.89022 | 1379.956665 | 71606.25493731629 | 0.0003166580282 |
| 005_shared_case | pass | 52.13808 | 1503.011475 | 78364.132524468 | 0.0003290738969 |
| 006_staged_priority | pass | 53.98974 | 1450.74353 | 78325.2659913822 | 0.0003292072797 |
| 007_staged_case | pass | 50.76756 | 1651.973267 | 83866.65195081853 | 0.0003589953412 |
| 008_shared_900 | pass | 52.13808 | 1503.011475 | 78364.132524468 | 0.0004387559893 |
| 009_staged_900 | pass | 50.76756 | 1651.973267 | 83866.65195081853 | 0.0004786512291 |
| 010_shared_1600 | pass | 50.72382 | 1615.485962 | 81943.61914901485 | 0.0002322194923 |
| 011_staged_1600 | pass | 50.76756 | 1651.973267 | 83866.65195081853 | 0.0002692533017 |
| 012_aligned_900 | pass | 56.45376 | 1311.40979 | 74034.0135463104 | 0.0003884203616 |
| 013_normtree | pass | 50.301 | 1499.21936 | 75412.23302736001 | 0.0003081332543 |
| 014_packed_round | pass | 53.98974 | 1451.885864 | 78386.94030703536 | 0.0004414863361 |
| 015_parallel_diff | pass | 54.8208 | 1258.278198 | 68979.8174369184 | 0.0004002646601 |
| 016_aligned_unconstrained | pass | 51.4674 | 1995.999634 | 102728.9115629316 | 0.0003059029114 |
| 017_shared_unconstrained | pass | 47.37042 | 1998.141602 | 94652.80690621285 | 0.0003318626841 |
| 018_parallel_unconstrained | pass | 49.7907 | 1926.210693 | 95907.3787519551 | 0.000339766877 |
| 019_packed_priority | pass | 53.34822 | 1333.746216 | 71152.98655533552 | 0.0004451620334 |
| 020_sentinel | pass | 52.29846 | 1254.09729 | 65587.3569571734 | 0.0003309532767 |
| 021_sentinel_priority | pass | 51.99228 | 1094.322144 | 56896.30332104832 | 0.0003195773752 |
| 022_parallel_exponent | failed | - | - | - | - |
| 023_prefix | pass | 53.61066 | 1240.94751 | 66528.0150364566 | 0.0003630670253 |
| 024_prefix_sentinel | pass | 52.54632 | 1309.072266 | 68786.93019236112 | 0.0003252348397 |
| 025_prefix_priority | pass | 54.25218 | 1159.015625 | 62879.1243103125 | 0.0002902904234 |
| 026_round_prefix | pass | 55.06866 | 1193.146729 | 65704.99154941314 | 0.0003270788293 |
| 027_round_sentinel | pass | 52.0506 | 1208.9552 | 62926.84353312001 | 0.0003504612541 |
| 028_round_priority | pass | 53.5815 | 1201.479736 | 64377.086474484 | 0.0002891568874 |
| 029_parallel_exponent_fixed | pass | 56.10384 | 1295.787964 | 72698.68060618176 | 0.0003068074875 |

## Joint ranking

Rank distinct RTL by area times maximum input-to-output delay. Multiple mapping targets for identical RTL do not count as distinct top-three designs.

1. `021_sentinel_priority`: 51.99228 um^2, 1094.322144 ps, ADP 56896.303.
2. `025_prefix_priority`: 54.25218 um^2, 1159.015625 ps, ADP 62879.124.
3. `027_round_sentinel`: 52.05060 um^2, 1208.955200 ps, ADP 62926.844.

## Comparison and limits

Paper adder baseline: 58 um^2 / 1610 ps. Optimized extremes: 49 um^2 and 1043 ps; not assumed to be one design. These are the adder results, not the multiplier results.

This experiment uses post-mapping ASAP7 RVT TT timing with no wire parasitics, a 10 ps input transition and 3.898 fF output load. Matching width and operation does not establish identical synthesis constraints. Power is vectorless. Canonical NaNs are used; no exception flag interface is implemented.

All passing runs verify both RTL and mapped gates on 3,359,296 directed/random pairs. This is not exhaustive binary16 verification. The independent Python model passes 851,968 checks against native binary16 packing.

See `docs/contract.md` and `campaign.py` for design hypotheses, numeric behavior and constraints. Frozen run metrics and stage logs are the measurement evidence. Failed runs are excluded. Fresh top-three replay is a separate acceptance gate.

## Current result

Best joint design area / paper minimum area = 1.061; delay / paper minimum delay = 1.049. These ratios compare separate paper extrema, not a joint reference point.

Relative to the exact fixed-point baseline: area reduction 58.55%, delay reduction 48.16%, ADP reduction 78.51%. Mapping targets differ; final STA loading is identical.

The best joint result is close to the paper extrema (approximately 6.1% greater area and 4.9% greater delay), not an improvement over those extrema. The requested similar-PPA exploration is achieved with the comparison limitations above; the 1000 ps mapping target remains unmet.

## Pareto frontier

- `017_shared_unconstrained`: 47.37042 um^2 / 1998.141602 ps.
- `018_parallel_unconstrained`: 49.79070 um^2 / 1926.210693 ps.
- `013_normtree`: 50.30100 um^2 / 1499.219360 ps.
- `004_shared_priority`: 51.89022 um^2 / 1379.956665 ps.
- `021_sentinel_priority`: 51.99228 um^2 / 1094.322144 ps.

## Optimization process

- Exact fixed-point baseline: transparent arithmetic, but wide shifts and normalization cost area.
- Compact alignment: keep significand plus guard/round/sticky bits; verify cancellation and subnormals.
- Shared add/subtract: reduce duplicate arithmetic, with a measurable area/delay tradeoff.
- Staged sticky shift: combine discarded-bit reduction with the barrel shifter.
- Normalization tree: compare incremental shifts with leading-bit decode; area improved but delay regressed.
- Packed rounding: carry directly into the encoded exponent/fraction.
- Parallel exponent differences: avoid waiting for operand swap before computing alignment distance.
- ABC load/drive ablation: smaller unconstrained mapping slowed final STA markedly; keep the same STA load in both cases.
- Sentinel normalization: fold the subnormal shift limit into leading-bit detection.
- Prefix arithmetic and rounding: test explicit parallel carry structures against operator mapping.

Each run is functionally checked before ranking. Runs 001 (reserved identifier) and 022 (missing token separator after endcase) are retained compile failures, not successful evaluations. Run 029 validates the latter correction.

## Fresh replay

Current top-three replay complete: True.
- `021_sentinel_priority` -> `replay_20260927_153134_021_sentinel_priority`: exact.
- `025_prefix_priority` -> `replay_20260927_153202_025_prefix_priority`: exact.
- `027_round_sentinel` -> `replay_20260927_153244_027_round_sentinel`: exact.
