"""Aggregate A/B attempt records into the selected data root's summary.json.

Run from exp/ab_test/:  python3 analysis/analyze.py
Reads full harness outputs or the packed evidence manifest. Full outputs are
checked with standalone iverilog; the manifest carries those checked verdicts.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AB_RESULTS", ROOT.parent / "results")).resolve()  # local run archive by default
sys.path.insert(0, str(ROOT.parent))
from src.grade.pareto import frontier, top3_pdap, eligible  # noqa: E402
from src.agent.parse import summarize as summarize_streams  # noqa: E402

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-6-sol"]
# group label -> (run directory, arm)
GROUPS = {
    "baseline": ("single", "notools"),
    "single-round": ("single", "tools"),
    "multi-round": ("multi", "tools"),
}


def syntax_ok(path):
    if not path.exists():
        return False
    r = subprocess.run(["iverilog", "-g2012", "-o", "/dev/null", str(path)],
                       capture_output=True, text=True)
    return r.returncode == 0


def api_error(directory):
    """True when any round's CLI stream ended in an API error (e.g. a 429 usage limit)."""
    for f in directory.glob("round*.jsonl"):
        for line in f.read_text().splitlines():
            if '"type":"result"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "result" and (ev.get("is_error") or ev.get("api_error_status")):
                return True
    return False


def usage_from_streams(directory):
    """(cost_usd, usage) re-derived from the saved round streams.

    result.json from runs before parse.summarize stopped summing session totals
    over-counts resumed rounds, so the raw streams are authoritative.
    """
    rounds = sorted(directory.glob("round*.jsonl"), key=lambda f: int(f.stem[len("round"):]))
    totals = summarize_streams([f.read_text() for f in rounds])
    return totals["cost_usd"], totals["usage"]


def load(model, group):
    """Merge runs of one group: per (problem, rep) keep the newest attempt without API errors."""
    manifest = DATA / "attempts.jsonl"
    if manifest.is_file():
        rows = []
        for line in manifest.read_text().splitlines():
            entry = json.loads(line)
            if entry["model"] != model or entry["group"] != group:
                continue
            row = entry["record"]
            row["_dir"] = DATA / entry["attempt_dir"]
            row["_run"] = DATA / entry["run"]
            row["_api_error"] = False  # Only clean attempts enter the manifest.
            row["_syntax_ok_rounds"] = entry["syntax_ok_rounds"]
            rows.append(row)
        rows.sort(key=lambda r: (r["problem"], r["rep"]))
        return (rows[0]["_run"] if rows else None), rows
    sub, arm = GROUPS[group]
    runs = sorted((DATA / model / sub).glob("*/"))
    if not runs:
        return None, []
    # multi-round has no repetitions: one attempt per problem, the earliest clean one
    # (chronological, never outcome-based). Other groups keep one attempt per (problem, rep).
    single = group == "multi-round"
    chosen = {}
    for run in runs:
        for f in sorted(run.glob(f"*__{arm}__*/result.json"), key=lambda f: int(f.parent.name.rsplit("__", 1)[1])):
            row = json.loads(f.read_text())
            row["_dir"], row["_run"] = f.parent, run
            row["_api_error"] = api_error(f.parent)
            row["cost_usd"], row["usage"] = usage_from_streams(f.parent)
            key = row["problem"] if single else (row["problem"], row["rep"])
            if key not in chosen or (chosen[key]["_api_error"] and not row["_api_error"]):
                chosen[key] = row
            elif not single and not row["_api_error"]:
                chosen[key] = row
    return runs[0], sorted(chosen.values(), key=lambda r: (r["problem"], r["rep"]))


def summarize(model, group, problems):
    run, rows = load(model, group)
    if not rows:
        return None
    snaps = syn = 0
    tool_calls, cost, rounds_done = {}, 0.0, []
    by_problem = {p: [] for p in problems}
    for row in rows:
        rounds_done.append(row.get("rounds", 0))
        cost += row.get("cost_usd") or 0
        for k, v in row.get("tool_calls", {}).items():
            tool_calls[k] = tool_calls.get(k, 0) + v
        n = max(row.get("rounds", 0), len(row["per_round"]), 1)
        for i in range(1, n + 1):
            snaps += 1
            if "_syntax_ok_rounds" in row:
                syn += row["_syntax_ok_rounds"][str(i)]
            else:
                syn += syntax_ok(row["_dir"] / f"round{i}.v")
        by_problem.setdefault(row["problem"], []).append(row)
    attempts = len(rows)
    func_problems, pass1, any_pass, cands, top3 = 0, [], [], {}, {}
    pass3 = []
    for p, prow in by_problem.items():
        if not prow:
            continue
        grades = [dict(g, problem=p, rep=r["rep"], _run=r["_run"]) for r in prow for g in r["per_round"]]
        func_problems += any(g.get("passed") for g in grades)
        if len(prow) == 3:
            pass3.append(any(bool(r.get("final_round", {}).get("passed")) for r in prow))
        # pass@1: per attempt, the delivered (final) snapshot is functionally correct.
        pass1.append(sum(bool(r.get("final_round", {}).get("passed")) for r in prow) / len(prow))
        any_pass.append(sum(any(g.get("passed") for g in r["per_round"]) for r in prow) / len(prow))
        front = frontier(grades)
        cands[p] = len(front)
        top3[p] = [dict(problem=p, rep=c["rep"], round=c["round"], pdap=c["pdap"],
                        area_um2=c["area_um2"], power_w=c["power_w"], delay_ns=c["delay_ns"],
                        solution_path=str((c["_run"] / c["solution_path"]).relative_to(DATA)))
                   for c in top3_pdap(grades)]
    probs = [p for p in by_problem if by_problem[p]]
    return dict(
        model=model, group=group, runs=sorted({str(r["_run"].relative_to(DATA)) for r in rows}), attempts=attempts,
        api_error_attempts=sum(r["_api_error"] for r in rows),
        problems=len(probs), snapshots=snaps, syntax_ok=syn, syntax_rate=syn / snaps if snaps else 0,
        function_problems=func_problems,
        expected_problems=len(problems), complete_triplets=len(pass3),
        pass_at_3=sum(pass3) / len(problems) if len(pass3) == len(problems) else None,
        pass_at_1=sum(pass1) / len(pass1), any_round_pass=sum(any_pass) / len(any_pass),
        mean_candidates=sum(cands.values()) / len(probs), candidates=cands, top3=top3,
        mean_rounds=sum(rounds_done) / attempts, cost_usd=cost, tool_calls=tool_calls,
        tool_use_attempts=sum(1 for r in rows if any(k not in ("Write", "Read", "Edit")
                                                    for k in r.get("tool_calls", {}))),
        timeouts=sum(bool(r.get("timed_out")) for r in rows),
        harness_failures=sum(1 for r in rows if str(r.get("failure", "")).startswith(("harness", "model", "timeout"))),
    )


def combined(model, problems):
    """Three groups compared together: per problem, frontier over all groups' candidates."""
    out = {}
    for p in problems:
        cands = []
        for group in GROUPS:
            run, rows = load(model, group)
            for r in rows:
                if r["problem"] != p:
                    continue
                for g in r["per_round"]:
                    cands.append(dict(g, group=group, rep=r["rep"], run=r["_run"]))
        front = frontier(cands)
        top = top3_pdap(cands)
        out[p] = dict(
            frontier_groups=[c["group"] for c in front],
            top3=[dict(group=c["group"], rep=c["rep"], round=c["round"], pdap=c["pdap"],
                       area_um2=c["area_um2"], power_w=c["power_w"], delay_ns=c["delay_ns"],
                       fmax_mhz=c["fmax_mhz"],
                       solution_path=str((c["run"] / c["solution_path"]).relative_to(DATA)))
                  for c in top],
            best_pdap={g: min((c["power_w"] * c["delay_ns"] * c["area_um2"] for c in cands
                               if c["group"] == g and eligible(c)), default=None) for g in GROUPS},
        )
    return out


def main():
    problems = sorted(p.parent.name for p in (ROOT.parent / "problems").rglob("design_description.txt"))
    summary = {"models": {}}
    for model in MODELS:
        groups = {g: s for g in GROUPS if (s := summarize(model, g, problems))}
        summary["models"][model] = dict(groups=groups, combined=combined(model, problems))
    (DATA / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({m: {g: {k: v for k, v in s.items() if k not in ("candidates", "top3")}
                          for g, s in d["groups"].items()} for m, d in summary["models"].items()},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
