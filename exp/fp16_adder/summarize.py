"""Rank verified original binary16 adders and render evidence-based reports."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def records():
    return [json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('[0-9][0-9][0-9]_*/metrics.json'))]

def ranked():
    valid=[r for r in records() if r['status']=='pass' and r.get('operation')=='add' and r.get('format')=='binary16 E5M10']
    unique={}
    for r in sorted(valid,key=lambda r:r['adp_um2_ps']): unique.setdefault(r['rtl_sha256'],r)
    return valid,list(unique.values())

def main():
    valid,distinct=ranked(); top=distinct[:3]
    (ROOT/'top3.json').write_text(json.dumps(top,indent=2),encoding='utf-8')
    lines=['# FP16 adder experiment report','','Target: RTLScout Section 7.1, binary16 addition (E5M10), local module `fpadd_fp16`.','',f'Completed passing evaluations: {len(valid)}; distinct passing RTL snapshots: {len(distinct)}.','',
    '| Run | Status | Area (um^2) | Delay (ps) | ADP | Estimated power (W) |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in records():
        lines.append(f"| {r['name']} | {r['status']} | {r.get('area_um2','-')} | {r.get('delay_ps','-')} | {r.get('adp_um2_ps','-')} | {r.get('estimated_power_w','-')} |")
    lines+=['','## Joint ranking','','Rank distinct RTL by area times maximum input-to-output delay. Multiple mapping targets for identical RTL do not count as distinct top-three designs.','']
    for i,r in enumerate(top,1): lines.append(f"{i}. `{r['name']}`: {r['area_um2']:.5f} um^2, {r['delay_ps']:.6f} ps, ADP {r['adp_um2_ps']:.3f}.")
    lines+=['','## Comparison and limits','','Paper adder baseline: 58 um^2 / 1610 ps. Optimized extremes: 49 um^2 and 1043 ps; not assumed to be one design. These are the adder results, not the multiplier results.','',
    'This experiment uses post-mapping ASAP7 RVT TT timing with no wire parasitics, a 10 ps input transition and 3.898 fF output load. Matching width and operation does not establish identical synthesis constraints. Power is vectorless. Canonical NaNs are used; no exception flag interface is implemented.','',
    'All passing runs verify both RTL and mapped gates on 3,359,296 directed/random pairs. This is not exhaustive binary16 verification. The independent Python model passes 851,968 checks against native binary16 packing.','',
    'See `docs/contract.md` and `campaign.py` for design hypotheses, numeric behavior and constraints. Frozen run metrics and stage logs are the measurement evidence. Failed runs are excluded. Fresh top-three replay is a separate acceptance gate.','']
    if top:
        r=top[0]; lines+=['## Current result','',f"Best joint design area / paper minimum area = {r['area_um2']/49:.3f}; delay / paper minimum delay = {r['delay_ps']/1043:.3f}. These ratios compare separate paper extrema, not a joint reference point.",'']
    if top:
        baseline=next(r for r in valid if r['name']=='000_exact_baseline')
        winner=top[0]
        lines += [f"Relative to the exact fixed-point baseline: area reduction {(1-winner['area_um2']/baseline['area_um2'])*100:.2f}%, delay reduction {(1-winner['delay_ps']/baseline['delay_ps'])*100:.2f}%, ADP reduction {(1-winner['adp_um2_ps']/baseline['adp_um2_ps'])*100:.2f}%. Mapping targets differ; final STA loading is identical.", '',
        'The best joint result is close to the paper extrema (approximately 6.1% greater area and 4.9% greater delay), not an improvement over those extrema. The requested similar-PPA exploration is achieved with the comparison limitations above; the 1000 ps mapping target remains unmet.', '']
    frontier=[r for r in valid if not any(q['area_um2']<=r['area_um2'] and q['delay_ps']<=r['delay_ps'] and (q['area_um2']<r['area_um2'] or q['delay_ps']<r['delay_ps']) for q in valid)]
    lines+=['## Pareto frontier','']
    seen=set()
    for r in sorted(frontier,key=lambda r:r['area_um2']):
        key=(r['area_um2'],r['delay_ps'])
        if key in seen: continue
        seen.add(key)
        lines.append(f"- `{r['name']}`: {r['area_um2']:.5f} um^2 / {r['delay_ps']:.6f} ps.")
    lines+=['','## Optimization process','',
    '- Exact fixed-point baseline: transparent arithmetic, but wide shifts and normalization cost area.',
    '- Compact alignment: keep significand plus guard/round/sticky bits; verify cancellation and subnormals.',
    '- Shared add/subtract: reduce duplicate arithmetic, with a measurable area/delay tradeoff.',
    '- Staged sticky shift: combine discarded-bit reduction with the barrel shifter.',
    '- Normalization tree: compare incremental shifts with leading-bit decode; area improved but delay regressed.',
    '- Packed rounding: carry directly into the encoded exponent/fraction.',
    '- Parallel exponent differences: avoid waiting for operand swap before computing alignment distance.',
    '- ABC load/drive ablation: smaller unconstrained mapping slowed final STA markedly; keep the same STA load in both cases.',
    '- Sentinel normalization: fold the subnormal shift limit into leading-bit detection.',
    '- Prefix arithmetic and rounding: test explicit parallel carry structures against operator mapping.',
    '', 'Each run is functionally checked before ranking. Runs 001 (reserved identifier) and 022 (missing token separator after endcase) are retained compile failures, not successful evaluations. Run 029 validates the latter correction.', '']
    replay=ROOT/'replay_verification.json'
    if replay.exists():
        evidence=json.loads(replay.read_text())
        current={r['name'] for r in top}
        verified={r['original'] for r in evidence}
        lines+=['## Fresh replay','',f'Current top-three replay complete: {current <= verified}.']
        for r in evidence: lines.append(f"- `{r['original']}` -> `{r['replay']}`: {r['agreement']}.")
        lines.append('')
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'passing':len(valid),'distinct':len(distinct),'top3':[r['name'] for r in top]}))
if __name__=='__main__': main()
