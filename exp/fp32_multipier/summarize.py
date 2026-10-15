"""Summarize the reviewed experiment fixtures and optional fresh replays."""
import csv
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for p in sorted((ROOT/'runs').glob('*/metrics.json')):
    m=json.loads(p.read_text());m['artifact']=p.parent.relative_to(ROOT).as_posix()
    if 'estimated_power_w' not in m and (p.parent/'sta.log').exists():
        match=re.search(r'^Total\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)',(p.parent/'sta.log').read_text(),re.M)
        if match:m['estimated_power_w']=float(match.group(1))
    rows.append(m)
original=[r for r in rows if r['name'][:3].isdigit()]
passed=[r for r in original if r['status']=='pass']
ranked=sorted(passed,key=lambda r:(r['adp_um2_ps'],r['name']))
top=[];seen=set()
for r in ranked:
    if r['rtl_sha256'] not in seen:top.append(r);seen.add(r['rtl_sha256'])
    if len(top)==3:break
(ROOT/'top3.json').write_text(json.dumps(top,indent=2))
columns=['name','mode','mul','lzd','target_delay_ps','abc_constrained','status','area_um2','delay_ps','adp_um2_ps','estimated_power_w','checks','gate_checks','artifact']
with (ROOT/'results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
baseline=next(r for r in passed if r['name']=='001_baseline')
matched=next(r for r in passed if r['name']=='032_baseline_sized')
best=top[0]
def table(rs):
    out=['| Run | Architecture / multiplier | Area (um^2) | Delay (ps) | ADP (um^2 ps) | Estimated power (uW) | RTL / gate checks |','|---|---|---:|---:|---:|---:|---:|']
    for r in rs:
        power=f"{r['estimated_power_w']*1e6:.3f}" if 'estimated_power_w' in r else 'unavailable'
        out.append(f"| [{r['name']}]({r['artifact']}/metrics.json) | {r['mode']} / {r['mul']} | {r['area_um2']:.5f} | {r['delay_ps']:.6f} | {r['adp_um2_ps']:.2f} | {power} | {r['checks']:,} / {r['gate_checks']:,} |")
    return '\n'.join(out)
report=f'''# FP12 multiplier experiment report

Publication: [issue #77](https://github.com/yenhao-huang/npu-pynq/issues/77).
Original experiment: [issue #76](https://github.com/yenhao-huang/npu-pynq/issues/76).
Experiment date: 2026-09-25. The user-requested `fp32_multipier` directory name is preserved; the circuit is FP12.

## Results

{len(passed)} successful original evaluations cover {len({r['rtl_sha256'] for r in passed})} distinct RTL snapshots. Failed attempts and recovery of STA integration errors are excluded from that count.
Best joint result: **{best['area_um2']:.5f} um^2, {best['delay_ps']:.6f} ps, ADP {best['adp_um2_ps']:.2f} um^2 ps**.
Against the initial baseline ({baseline['area_um2']:.5f} um^2, {baseline['delay_ps']:.6f} ps), ADP improves {100*(1-best['adp_um2_ps']/baseline['adp_um2_ps']):.2f}%, delay improves {100*(1-best['delay_ps']/baseline['delay_ps']):.2f}%, and area changes by {100*(best['area_um2']/baseline['area_um2']-1):+.2f}%.
Against the baseline using the same constrained ABC mapping and 700 ps target (032: {matched['area_um2']:.5f} um^2, {matched['delay_ps']:.6f} ps), area improves {100*(1-best['area_um2']/matched['area_um2']):.2f}%, delay {100*(1-best['delay_ps']/matched['delay_ps']):.2f}%, and ADP {100*(1-best['adp_um2_ps']/matched['adp_um2_ps']):.2f}%.

The original top-three replays passed all 16,777,216 input pairs in both RTL and mapped-netlist simulation and reproduced area/delay exactly. Publication worktree replays are verified separately with `reproduce.py` and `audit.py`; their generated logs remain local.

## Measurement contract

See [contract.md](docs/contract.md). E5M6 has one sign bit, five exponent bits, six fraction bits and bias 15. The design supports gradual underflow, ties-to-even rounding, signed zero and overflow to infinity. All NaNs and infinity times zero produce canonical positive qNaN 0x7e0; payloads and exception flags are not retained.

The flow is RTL generation or a frozen source file, Verilator compilation/simulation, Yosys/ABC mapping, mapped-netlist Verilator simulation, and Docker OpenROAD STA. `status=pass` means functional validation and measurement succeeded, not that the requested mapping delay was met.

ASAP7 7.5T RVT TT NLDM combinational AO/INVBUF/OA/SIMPLE libraries are used. Original source hashes are in [pdk-sources.json](docs/pdk-sources.json). OpenROAD loads the technology/cell LEFs. `report_units` confirms 1 ps and 1 fF. Input slew is 10 ps, output load 3.898 fF, and IO delays are zero. There are no interconnect parasitics: these are **post-mapping estimates**, not routed signoff, FPGA Fmax or silicon measurements.

Constrained ABC runs use BUFx2 as the input driver and 3.898 fF load, enabling buffering/sizing. OpenSTA uses the same explicit slew/load for every run. Vectorless estimated power depends on each run's virtual-clock period; do not compare different target periods as equal-frequency power results. Power is not part of ranking.

Versions and immutable OpenROAD image digest are in [environment.json](environment.json). The image reports OpenROAD version `unknown`; its binary SHA-256 and image digest identify the executable. Verilator 5.006 and Yosys 0.23/ABC 1.01 are reused from the existing helper container.

## Top three distinct RTL designs

ADP is mapped cell area times worst input-to-output delay. Each distinct RTL contributes only its best setting, so multiple synthesis settings of the same circuit cannot occupy all three positions.

{table(top)}

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

{table(passed)}

## Rejected attempts

'''
for r in original:
    if r['status']!='pass':report+=f"- {r['name']}: {r['status']} — {r.get('error','unfinished')}\n"
report+='\nInitial STA integration required loading technology LEF and adapting to the current OpenSTA reporting API. Recovered STA measurements reuse already verified mapped netlists; recovery was not counted as additional architecture experiments. The primary workspace retains the earlier diagnostic logs.\n'
(ROOT/'report.md').write_text(report,encoding='utf-8')
print(json.dumps({'passing_experiments':len(passed),'top3':[r['name'] for r in top]}))
