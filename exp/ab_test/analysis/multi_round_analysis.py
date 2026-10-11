"""Why multi-round beats single-round: per-round cost, best-of-k curves, per-problem deltas.

Report.md compares whole arms. This answers a narrower question: single-round has the
same IC tools, so where does the multi-round advantage come from, and what does a round
cost relative to a single-round attempt?

Run from exp/ab_test/:  python3 analysis/multi_round_analysis.py  -> evidence/multi_round_analysis.json
"""
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AB_RESULTS", ROOT.parent / "results")).resolve()  # local run archive by default
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from analyze import load, MODELS  # noqa: E402
from src.grade.pareto import eligible  # noqa: E402

MAX_ROUNDS = 5
TOP_N = 5


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def clean(rows):
    """Attempts analyze.py selected, minus any whose CLI stream ended in an API error."""
    return [r for r in rows if not r["_api_error"]]


def pdap_by_round(row):
    """round number -> PDAP, for the rounds of one attempt that produced a valid candidate."""
    return {g["round"]: g["power_w"] * g["delay_ns"] * g["area_um2"]
            for g in row["per_round"] if eligible(g)}


def candidate(row, number):
    """The full measurement of one round, for reporting area/power/delay alongside PDAP."""
    for g in row["per_round"]:
        if g["round"] == number:
            return g
    return None


def round_usage(row):
    """round number -> (cost_usd, input_tokens, output_tokens), read from the CLI stream.

    Claude reports a billed cost; Codex reports tokens only, so its cost stays 0.
    Input counts every token the request was charged for, cached reads included.
    """
    usage = {}
    streams = sorted(row["_dir"].glob("round*.jsonl"))
    if not streams:
        raise FileNotFoundError(
            f"Per-round usage needs the archived CLI streams under {row['_dir']}"
        )
    for path in streams:
        cost = inputs = outputs = 0
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "result" and "total_cost_usd" in event:        # Claude
                cost = event["total_cost_usd"] or 0
                u = event.get("usage") or {}
                inputs = sum((u.get(k) or 0) for k in
                             ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                outputs = u.get("output_tokens") or 0
            elif event.get("type") == "turn.completed":                            # Codex
                u = event.get("usage") or {}
                inputs += u.get("input_tokens") or 0
                outputs += u.get("output_tokens") or 0
        usage[int(path.stem.replace("round", ""))] = (cost, inputs, outputs)
    return usage


TYPE_WORDS = {"wire", "reg", "logic", "bit", "signed", "unsigned"}


def input_ports(source):
    """Every driven-from-outside port, including `input wire A, B, C` on one line."""
    names = set()
    for keyword in re.finditer(r"\b(?:input|inout)\b", source):
        tail = re.split(r"[;)]|\b(?:input|output|inout)\b", source[keyword.end():])[0]
        for item in re.sub(r"\[[^\]]*\]", " ", tail).split(","):
            words = [w for w in re.findall(r"[A-Za-z_]\w*", item) if w not in TYPE_WORDS]
            if words:
                names.add(words[0])
    return names


def derived_clocks(source):
    """Edge-sensitive signals that are not module inputs, i.e. internally generated clocks.

    OpenROAD times only the primary clock here, so a design that clocks state off a
    divided or rippled signal gets an optimistic delay and power figure. This is a
    syntactic flag for review, not a verdict.
    """
    edges = {signal for block in re.findall(r"@\s*\(([^)]*)\)", source)
             for signal in re.findall(r"\b(?:pos|neg)edge\s+([A-Za-z_]\w*)", block)}
    return sorted(edges - input_ports(source))


def api_cost_estimate(rows, units):
    """GPT-6 Sol Standard API price bounds, USD per attempt or round (2026-10-11).

    Cached reads use their own rate. The run archive does not identify which
    individual requests crossed the 272K long-context threshold, so apply the
    short and long rates to all tokens as illustrative bounds, not a bill.
    """
    usage = {key: sum(row.get("usage", {}).get(key, 0) or 0 for row in rows)
             for key in ("input_tokens", "cache_read_input_tokens", "output_tokens")}
    def price(input_rate, cached_rate, output_rate):
        return (usage["input_tokens"] * input_rate
                + usage["cache_read_input_tokens"] * cached_rate
                + usage["output_tokens"] * output_rate) / 1_000_000 / units
    return dict(short_context_usd=price(2, 0.2, 10),
                long_context_usd=price(4, 0.4, 15))


def cost_table(baseline, single, multi, model):
    """Mean cost and tokens per baseline/single attempt and multi-round round."""
    baseline_values = [u for row in baseline for u in round_usage(row).values()]
    base = [u for row in single for u in round_usage(row).values()]
    mean = lambda rows, i: sum(u[i] for u in rows) / len(rows)
    by_round = {}
    for row in multi:
        for number, u in round_usage(row).items():
            by_round.setdefault(number, []).append(u)
    every = [u for group in by_round.values() for u in group]
    ratio = lambda rows, i: mean(rows, i) / mean(base, i) if mean(base, i) else None
    result = dict(
        baseline_attempt=dict(n=len(baseline_values), cost_usd=mean(baseline_values, 0),
                              input_tokens=mean(baseline_values, 1),
                              output_tokens=mean(baseline_values, 2)),
        single_attempt=dict(n=len(base), cost_usd=mean(base, 0),
                            input_tokens=mean(base, 1), output_tokens=mean(base, 2)),
        multi_round_mean=dict(n=len(every), cost_usd=mean(every, 0),
                              input_tokens=mean(every, 1), output_tokens=mean(every, 2),
                              cost_ratio=ratio(every, 0), input_ratio=ratio(every, 1),
                              output_ratio=ratio(every, 2)),
        per_round={n: dict(n=len(g), cost_usd=mean(g, 0), input_tokens=mean(g, 1),
                           output_tokens=mean(g, 2), cost_ratio=ratio(g, 0),
                           input_ratio=ratio(g, 1), output_ratio=ratio(g, 2))
                   for n, g in sorted(by_round.items())})
    if model == "gpt-6-sol":
        result["baseline_attempt"]["api_estimate"] = api_cost_estimate(baseline, len(baseline_values))
        result["single_attempt"]["api_estimate"] = api_cost_estimate(single, len(base))
        result["multi_round_mean"]["api_estimate"] = api_cost_estimate(multi, len(every))
    return result


def best_of_k(single_best, multi_rounds, problems):
    """Both curves in units of the expected best of one single-round attempt.

    Single-round draws are exchangeable, so best-of-k averages over every k-subset of the
    repetitions. Multi-round rounds are sequential, so k is the prefix of rounds 1..k.
    """
    reference = {p: geomean(single_best[p]) for p in problems}
    independent = {}
    for k in range(1, 4):
        values = [geomean([min(sub) for sub in itertools.combinations(single_best[p], k)]) / reference[p]
                  for p in problems if len(single_best[p]) >= k]
        independent[k] = dict(geomean=geomean(values), n=len(values))
    conditioned = {}
    for k in range(1, MAX_ROUNDS + 1):
        values = [min(v for n, v in multi_rounds[p].items() if n <= k) / reference[p]
                  for p in problems if any(n <= k for n in multi_rounds[p])]
        conditioned[k] = dict(geomean=geomean(values), n=len(values))
    return dict(single_round=independent, multi_round=conditioned)


def main():
    out = {}
    for model in MODELS:
        baseline = clean(load(model, "baseline")[1])
        single, multi = clean(load(model, "single-round")[1]), clean(load(model, "multi-round")[1])

        single_best, single_pick = {}, {}
        for row in single:
            for number, value in pdap_by_round(row).items():
                single_best.setdefault(row["problem"], []).append(value)
                if value < single_pick.get(row["problem"], (math.inf,))[0]:
                    single_pick[row["problem"]] = (value, candidate(row, number), row)
        multi_rounds, multi_pick = {}, {}
        for row in multi:
            rounds = pdap_by_round(row)
            if not rounds:
                continue
            multi_rounds[row["problem"]] = rounds
            number = min(rounds, key=rounds.get)
            multi_pick[row["problem"]] = (rounds[number], candidate(row, number), row, number)
        problems = sorted(set(single_pick) & set(multi_pick))

        deltas = []
        for p in problems:
            s, m = single_pick[p], multi_pick[p]
            source = m[2]["_dir"] / f"round{m[3]}.v"
            deltas.append(dict(
                problem=p, ratio=m[0] / s[0], best_round=m[3], rounds_measured=len(multi_rounds[p]),
                single={k: s[1][k] for k in ("area_um2", "power_w", "delay_ns")},
                multi={k: m[1][k] for k in ("area_um2", "power_w", "delay_ns")},
                derived_clocks=derived_clocks(source.read_text()) if source.exists() else [],
                solution_path=str(source.relative_to(DATA))))
        deltas.sort(key=lambda d: d["ratio"])
        suspect = [d["problem"] for d in deltas if d["derived_clocks"]]
        trusted = [d["ratio"] for d in deltas if not d["derived_clocks"]]

        # How often do three fresh sessions on the same specification land on one design?
        complete = [p for p, v in single_best.items() if len(v) == 3]
        identical = [p for p in complete
                     if max(single_best[p]) - min(single_best[p]) < 1e-9 * max(single_best[p])]

        out[model] = dict(
            cost=cost_table(baseline, single, multi, model),
            best_of_k=best_of_k(single_best, multi_rounds, problems),
            collapse=dict(identical=len(identical), complete=len(complete), problems=sorted(identical)),
            paired=dict(geomean=geomean([d["ratio"] for d in deltas]), n=len(deltas),
                        geomean_excluding_derived_clocks=geomean(trusted) if trusted else None,
                        n_excluding=len(trusted), derived_clock_problems=suspect),
            top=deltas[:TOP_N])
    (DATA / "multi_round_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
