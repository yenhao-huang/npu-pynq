"""Paired best-PDAP comparisons vs baseline (incl. round-limited multi-round) and token totals.

Run from exp/ab_test/:  python3 analysis/extras.py   -> evidence/extras.json
"""
import json
import os
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AB_RESULTS", ROOT.parent / "results")).resolve()  # local run archive by default
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from analyze import load, GROUPS, MODELS  # noqa: E402
from src.grade.pareto import eligible, frontier  # noqa: E402

PROBLEMS = sorted(p.parent.name for p in (ROOT.parent / "problems").rglob("design_description.txt"))
TOKEN_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
              "output_tokens", "reasoning_output_tokens")


def grades(rows, problem, max_round=None):
    return [g for r in rows if r["problem"] == problem for g in r["per_round"]
            if max_round is None or g["round"] <= max_round]


def best(rows, problem, max_round=None):
    v = [g["power_w"] * g["delay_ns"] * g["area_um2"] for g in grades(rows, problem, max_round) if eligible(g)]
    return min(v) if v else None


def main():
    out = {}
    for model in MODELS:
        rows = {g: load(model, g)[1] for g in GROUPS}
        paired = {}
        for group, max_round in [("single-round", None), ("multi-round", None), ("multi-round", 3), ("multi-round", 1)]:
            ratios, win, tie, loss, base_only, group_only = [], 0, 0, 0, 0, 0
            for p in PROBLEMS:
                b, c = best(rows["baseline"], p), best(rows[group], p, max_round)
                if b is None or c is None:
                    base_only += b is not None and c is None
                    group_only += c is not None and b is None
                    continue
                ratio = c / b
                ratios.append(ratio)
                if math.isclose(ratio, 1, rel_tol=1e-6):
                    tie += 1
                elif ratio < 1:
                    win += 1
                else:
                    loss += 1
            key = group + ("" if max_round is None else f"@R<={max_round}")
            paired[key] = dict(geomean=round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 3),
                               n=len(ratios), win=win, tie=tie, loss=loss,
                               baseline_only=base_only, group_only=group_only,
                               mean_cands=round(sum(len(frontier(grades(rows[group], p, max_round)))
                                                    for p in PROBLEMS) / len(PROBLEMS), 2))
        tokens = {}
        for group in GROUPS:
            totals = {}
            for r in rows[group]:
                for k, v in r.get("usage", {}).items():
                    if k in TOKEN_KEYS:
                        totals[k] = totals.get(k, 0) + v
            totals["wall_h"] = round(sum(r.get("wall_s", 0) for r in rows[group]) / 3600, 1)
            tokens[group] = totals
        out[model] = dict(paired=paired, tokens=tokens)
    (DATA / "extras.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
