# FP12 multiplier experiment report

Publication: [issue #77](https://github.com/yenhao-huang/npu-pynq/issues/77).
Original experiment: [issue #76](https://github.com/yenhao-huang/npu-pynq/issues/76).
Experiment date: 2026-09-25. The user-requested `fp32_multipier` directory name is preserved; the circuit is FP12.

## Results

32 successful original evaluations cover 22 distinct RTL snapshots. Failed attempts and recovery of STA integration errors are excluded from that count.
Best joint result: **48.12858 um^2, 863.236816 ps, ADP 41546.36 um^2 ps**.
Against the initial baseline (46.48104 um^2, 1538.862671 ps), ADP improves 41.92%, delay improves 43.90%, and area changes by +3.54%.
Against the baseline using the same constrained ABC mapping and 700 ps target (032: 49.25124 um^2, 1296.354492 ps), area improves 2.28%, delay 33.41%, and ADP 34.93%.

The original top-three replays passed all 16,777,216 input pairs in both RTL and mapped-netlist simulation and reproduced area/delay exactly. Publication worktree replays are verified separately with `reproduce.py` and `audit.py`; their generated logs remain local.

## Measurement contract

See [contract.md](docs/contract.md). E5M6 has one sign bit, five exponent bits, six fraction bits and bias 15. The design supports gradual underflow, ties-to-even rounding, signed zero and overflow to infinity. All NaNs and infinity times zero produce canonical positive qNaN 0x7e0; payloads and exception flags are not retained.

The flow is RTL generation or a frozen source file, Verilator compilation/simulation, Yosys/ABC mapping, mapped-netlist Verilator simulation, and Docker OpenROAD STA. `status=pass` means functional validation and measurement succeeded, not that the requested mapping delay was met.

ASAP7 7.5T RVT TT NLDM combinational AO/INVBUF/OA/SIMPLE libraries are used. Original source hashes are in [pdk-sources.json](docs/pdk-sources.json). OpenROAD loads the technology/cell LEFs. `report_units` confirms 1 ps and 1 fF. Input slew is 10 ps, output load 3.898 fF, and IO delays are zero. There are no interconnect parasitics: these are **post-mapping estimates**, not routed signoff, FPGA Fmax or silicon measurements.

Constrained ABC runs use BUFx2 as the input driver and 3.898 fF load, enabling buffering/sizing. OpenSTA uses the same explicit slew/load for every run. Vectorless estimated power depends on each run's virtual-clock period; do not compare different target periods as equal-frequency power results. Power is not part of ranking.

Versions and immutable OpenROAD image digest are in [environment.json](environment.json). The image reports OpenROAD version `unknown`; its binary SHA-256 and image digest identify the executable. Verilator 5.006 and Yosys 0.23/ABC 1.01 are reused from the existing helper container.

## Top three distinct RTL designs

ADP is mapped cell area times worst input-to-output delay. Each distinct RTL contributes only its best setting, so multiple synthesis settings of the same circuit cannot occupy all three positions.

| Run | Architecture / multiplier | Area (um^2) | Delay (ps) | ADP (um^2 ps) | Estimated power (uW) | RTL / gate checks |
|---|---|---:|---:|---:|---:|---:|
| [034_onehot_fix](runs/034_onehot_fix/metrics.json) | onehot / operator | 48.12858 | 863.236816 | 41546.36 | 274.250 | 16,777,216 / 16,777,216 |
| [022_packed_priority](runs/022_packed_priority/metrics.json) | packed / operator | 47.15172 | 944.229614 | 44522.05 | 341.245 | 445,760 / 445,760 |
| [026_parallel_exp](runs/026_parallel_exp/metrics.json) | parallel / operator | 50.18436 | 971.649963 | 48761.63 | 393.490 | 16,777,216 / 16,777,216 |

The original screening count for rank 2 is shown above; its subsequent fresh full-domain replay passed 16,777,216 checks per simulation stage. Follow [reproduce.md](reproduce.md) to regenerate and verify all three.

## Hypothesis, result, decision

| Experiments | Hypothesis | Observation and decision |
|---|---|---|
| 000 / 001 | Start with unnormalized significands and dynamic shifting | Exhaustive simulation found smallest-subnormal squared incorrectly rounding to one. Clamp shifts above 14 to zero; corrected RTL and mapped simulation pass all pairs. |
| 002–004 | Fixed guard/sticky slices and narrower intermediate values | Area increased to 52.34220 from 46.48104; narrowing declarations alone did not help. 003 accidentally narrowed a sticky slice; preserve failure and fix in 004. |
| 005–015 | Sweep leading-bit encoding, multiplier structure, mapping targets | CSA sometimes reduces delay; effects are architecture dependent. Loose targets can map identically. ABC fast mode regressed strongly and was rejected. |
| 016 / 017 | Separate rounding paths or normalize inputs first | Separate rounding reached 1180.012085 ps at 56.33712 um^2; input normalization reduced area but delayed the joint improvement. |
| 018 | Round the packed exponent/fraction with one increment | Avoiding separate mantissa and exponent carry correction reached 45.79578 um^2 / 1167.081543 ps. Use as the next seed. |
| 019–025 | Add explicit mapping loads and sizing; sweep arithmetic | Priority encoding with packed rounding reached 47.15172 um^2 / 944.229614 ps. Dual rounding reached 924.997681 ps at higher area. |
| 026–032 | Compute exponent candidates in parallel; test nearby targets | The best parallel candidate reached 971.649963 ps. Same-condition baseline 032 separates RTL improvement from synthesis-setting effects. |
| 033 / 034 | Replace encoded normalization selection with one-hot masks | Correct a generator syntax failure in 033. 034 passes exhaustive RTL/gate tests and reaches the best ADP and delay: 48.12858 um^2 / 863.236816 ps. |

## Paper comparison and limitations

The comparison uses [RTLScout arXiv:2606.06530v2](https://arxiv.org/abs/2606.06530v2), Table 1, not a later revision. Its FP16 baseline is 121 um^2 / 1618 ps; its final minimum area is 79 um^2 and minimum delay 891 ps, attained by different Pareto points.
Our best FP12 candidate is below both of those absolute reference values under the declared mapping setup. This is a comparable PPA scale, **not a like-for-like FP16 reproduction or a claim to outperform the paper's circuit**. Precision, NaN handling, tool versions and unspecified paper IO constraints differ. The paper's 35% area and 45% delay reductions are relative to its own FP16 baseline; our actual relative results are stated separately above.

The exact integer Python oracle was also compared with an independent nearest-representable-grid search for 2,018,272 checks. C++ simulation uses integer products and binade threshold comparisons. Screening uses 445,760 directed/random checks with seed 123456789; finalist replays are exhaustive.

No NPU RTL, numeric model, register map, AXI interface, exported model format or board runtime changes. No Vivado, board or routed timing results are claimed.

## Complete successful experiment table

| Run | Architecture / multiplier | Area (um^2) | Delay (ps) | ADP (um^2 ps) | Estimated power (uW) | RTL / gate checks |
|---|---|---:|---:|---:|---:|---:|
| [001_baseline](runs/001_baseline/metrics.json) | baseline / operator | 46.48104 | 1538.862671 | 71527.94 | 225.071 | 16,777,216 / 16,777,216 |
| [002_slices](runs/002_slices/metrics.json) | slices / operator | 52.34220 | 1413.059814 | 73962.66 | 320.968 | 16,777,216 / 16,777,216 |
| [004_compact_fix](runs/004_compact_fix/metrics.json) | compact / operator | 52.34220 | 1413.059814 | 73962.66 | 320.968 | 16,777,216 / 16,777,216 |
| [005_baseline_case](runs/005_baseline_case/metrics.json) | baseline / operator | 48.82842 | 1576.399902 | 76973.12 | 248.787 | 445,760 / 445,760 |
| [006_compact_case](runs/006_compact_case/metrics.json) | compact / operator | 54.71874 | 1454.479736 | 79587.30 | 306.829 | 445,760 / 445,760 |
| [007_compact_csa](runs/007_compact_csa/metrics.json) | compact / csa | 54.76248 | 1288.315552 | 70551.35 | 287.179 | 445,760 / 445,760 |
| [008_compact_rows](runs/008_compact_rows/metrics.json) | compact / rows | 55.09782 | 1395.117554 | 76867.94 | 324.153 | 445,760 / 445,760 |
| [009_compact_split](runs/009_compact_split/metrics.json) | compact / split | 53.05662 | 1347.453491 | 71491.33 | 348.903 | 445,760 / 445,760 |
| [010_baseline_csa](runs/010_baseline_csa/metrics.json) | baseline / csa | 50.41764 | 1645.735718 | 82974.11 | 219.052 | 445,760 / 445,760 |
| [011_baseline_rows](runs/011_baseline_rows/metrics.json) | baseline / rows | 48.20148 | 1639.342163 | 79018.72 | 261.691 | 445,760 / 445,760 |
| [012_baseline_split](runs/012_baseline_split/metrics.json) | baseline / split | 47.32668 | 1625.396484 | 76924.62 | 313.533 | 445,760 / 445,760 |
| [013_compact_900](runs/013_compact_900/metrics.json) | compact / operator | 54.71874 | 1454.479736 | 79587.30 | 409.095 | 445,760 / 445,760 |
| [014_compact_1700](runs/014_compact_1700/metrics.json) | compact / operator | 54.71874 | 1454.479736 | 79587.30 | 216.593 | 445,760 / 445,760 |
| [015_compact_fast](runs/015_compact_fast/metrics.json) | compact / operator | 84.50568 | 2476.009766 | 209236.89 | 815.761 | 445,760 / 445,760 |
| [016_dual](runs/016_dual/metrics.json) | dual / operator | 56.33712 | 1180.012085 | 66478.48 | 248.123 | 16,777,216 / 16,777,216 |
| [017_prenorm](runs/017_prenorm/metrics.json) | prenorm / operator | 45.98532 | 1350.147827 | 62086.98 | 211.998 | 16,777,216 / 16,777,216 |
| [018_packed](runs/018_packed/metrics.json) | packed / operator | 45.79578 | 1167.081543 | 53447.41 | 173.103 | 16,777,216 / 16,777,216 |
| [019_packed_sized](runs/019_packed_sized/metrics.json) | packed / operator | 47.50164 | 1042.917603 | 49540.30 | 318.645 | 16,777,216 / 16,777,216 |
| [020_packed_csa](runs/020_packed_csa/metrics.json) | packed / csa | 50.15520 | 1022.600464 | 51288.73 | 342.708 | 445,760 / 445,760 |
| [021_packed_split](runs/021_packed_split/metrics.json) | packed / split | 47.10798 | 1150.929199 | 54217.95 | 372.647 | 445,760 / 445,760 |
| [022_packed_priority](runs/022_packed_priority/metrics.json) | packed / operator | 47.15172 | 944.229614 | 44522.05 | 341.245 | 445,760 / 445,760 |
| [023_prenorm_sized](runs/023_prenorm_sized/metrics.json) | prenorm / operator | 47.26836 | 1113.933838 | 52653.83 | 374.816 | 445,760 / 445,760 |
| [024_dual_sized](runs/024_dual_sized/metrics.json) | dual / operator | 58.17420 | 924.997681 | 53811.00 | 449.789 | 445,760 / 445,760 |
| [025_packed_rows](runs/025_packed_rows/metrics.json) | packed / rows | 48.46392 | 1158.063599 | 56124.30 | 344.580 | 445,760 / 445,760 |
| [026_parallel_exp](runs/026_parallel_exp/metrics.json) | parallel / operator | 50.18436 | 971.649963 | 48761.63 | 393.490 | 16,777,216 / 16,777,216 |
| [027_parallel_priority](runs/027_parallel_priority/metrics.json) | parallel / operator | 49.79070 | 1081.859863 | 53866.56 | 386.513 | 445,760 / 445,760 |
| [028_parallel_csa](runs/028_parallel_csa/metrics.json) | parallel / csa | 50.19894 | 976.077759 | 48998.07 | 356.101 | 445,760 / 445,760 |
| [029_packed_priority_900](runs/029_packed_priority_900/metrics.json) | packed / operator | 47.15172 | 944.229614 | 44522.05 | 265.418 | 445,760 / 445,760 |
| [030_packed_priority_1200](runs/030_packed_priority_1200/metrics.json) | packed / operator | 45.57708 | 1135.973145 | 51774.34 | 186.528 | 445,760 / 445,760 |
| [031_packed_priority_300](runs/031_packed_priority_300/metrics.json) | packed / operator | 47.15172 | 944.229614 | 44522.05 | 794.690 | 445,760 / 445,760 |
| [032_baseline_sized](runs/032_baseline_sized/metrics.json) | baseline / operator | 49.25124 | 1296.354492 | 63847.07 | 418.843 | 445,760 / 445,760 |
| [034_onehot_fix](runs/034_onehot_fix/metrics.json) | onehot / operator | 48.12858 | 863.236816 | 41546.36 | 274.250 | 16,777,216 / 16,777,216 |

## Rejected attempts

- 000_initial_baseline: failed — Initial terminal-observed exhaustive simulation failed at a=001 b=001 expected=000 actual=001 checks=4098. Archived original RTL; replaced rtl/baseline.sv with the verified 001 snapshot. No PPA measured.
- 003_compact: failed — exit=1: C:\Users\User\Desktop\agent_workspace\npu\npu_repo_in_pynq\exp\fp32_multipier\runs\003_compact\simulation.log
- 033_onehot: failed — f-string: single '}' is not allowed (architectures.py, line 127)

Initial STA integration required loading technology LEF and adapting to the current OpenSTA reporting API. Recovered STA measurements reuse already verified mapped netlists; recovery was not counted as additional architecture experiments. The primary workspace retains the earlier diagnostic logs.
