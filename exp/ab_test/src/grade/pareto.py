"""Exact Pareto filtering of measured candidates for the same problem."""
import math

OBJECTIVES = ("area_um2", "power_w", "delay_ns")

def eligible(candidate):
    """Require successful correctness and PPA evaluation with valid metrics."""
    if candidate.get("passed") is not True or candidate.get("ppa_ok") is not True:
        return False
    if candidate.get("passed_shipped") is not True:
        return False
    if candidate.get("passed_heldout") is not True and not candidate.get("heldout_skipped", False):
        return False
    return all(isinstance(candidate.get(key), (int, float))
               and not isinstance(candidate[key], bool)
               and math.isfinite(candidate[key]) and candidate[key] > 0
               for key in OBJECTIVES)

def dominates(a, b):
    return (all(a[key] <= b[key] for key in OBJECTIVES)
            and any(a[key] < b[key] for key in OBJECTIVES))

def frontier(candidates):
    """Preserve input order and tied candidates; all objectives are minimized."""
    valid = [candidate for candidate in candidates if eligible(candidate)]
    return [a for a in valid if not any(dominates(b, a) for b in valid)]

def top3_pdap(candidates):
    """Rank only the Pareto frontier; PDAP never affects frontier membership."""
    ranked = [dict(candidate, pdap=candidate["power_w"] * candidate["delay_ns"] * candidate["area_um2"])
              for candidate in frontier(candidates)]
    return sorted(ranked, key=lambda candidate: candidate["pdap"])[:3]


def grouped_frontiers(rows):
    """Compare rounds across attempts, separately by problem and by arm."""
    groups = {}
    for row in rows:
        problem = row["problem"]
        group = groups.setdefault(problem, {"candidates": [], "arms": {}})
        for grade in row.get("per_round", []):
            candidate = dict(grade, problem=problem, arm=row["arm"], rep=row["rep"])
            group["candidates"].append(candidate)
            group["arms"].setdefault(row["arm"], []).append(candidate)
    return {problem: {
        "frontier": frontier(group["candidates"]),
        "top3_pdap": top3_pdap(group["candidates"]),
        "top3_pdap_by_arm": {arm: top3_pdap(candidates) for arm, candidates in group["arms"].items()},
        "by_arm": {arm: frontier(candidates) for arm, candidates in group["arms"].items()},
        "evaluated_candidates": len(group["candidates"]),
        "eligible_candidates": sum(eligible(c) for c in group["candidates"]),
    } for problem, group in groups.items()}
