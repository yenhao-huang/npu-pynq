"""Render aggregate correctness and paired PDAP comparisons from summary.json."""
import json
import os
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AB_RESULTS", ROOT.parent / "results")).resolve()  # local run archive by default
GROUPS = [("baseline", "notools"), ("single-round", "tools"), ("multi-round", "tools")]
SHORT = {"claude-opus-5-5": "Opus 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "gpt-6-sol": "gpt-6-sol"}


def pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def pdap_comparison(combined, group):
    ratios = []
    wins = ties = losses = baseline_only = group_only = 0
    for candidate in combined.values():
        best = candidate["best_pdap"]
        baseline, current = best.get("baseline"), best.get(group)
        if baseline is None or current is None:
            baseline_only += baseline is not None and current is None
            group_only += current is not None and baseline is None
            continue
        ratio = current / baseline
        ratios.append(ratio)
        if math.isclose(ratio, 1.0, rel_tol=1e-6):
            ties += 1
        elif ratio < 1:
            wins += 1
        else:
            losses += 1
    geomean = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None
    return geomean, len(ratios), wins, ties, losses, baseline_only, group_only


def shell_ratio(old_combined, shell_combined):
    """Best shell-baseline PDAP over best no-shell baseline PDAP, per problem, geometric mean."""
    ratios = []
    for p, c in old_combined.items():
        base, shell = c["best_pdap"].get("baseline"), shell_combined.get(p, {}).get("best_pdap", {}).get("baseline")
        if base is not None and shell is not None:
            ratios.append(shell / base)
    return (math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None), len(ratios)


def main():
    s = json.loads((DATA / "summary.json").read_text())
    shell_path = DATA / "shell" / "summary.json"
    shell = json.loads(shell_path.read_text())["models"] if shell_path.exists() else {}
    out = []
    for model, data in s["models"].items():
        g = data["groups"]
        any_g = next(iter(g.values()))
        out.append(f"## {SHORT.get(model, model)}\n")
        budget = "無（Codex CLI 沒有預算參數）" if model.startswith("gpt") else "$3.00 / 次呼叫"
        out.append(f"**實驗設定**：模型 `{model}`｜effort `medium`｜題數 `{any_g['problems']}`"
                   f"｜每題獨立重複：baseline / single-round `3` 次，multi-round `1` 次｜每輪預算 {budget}\n")
        out.append("| 實驗組 | Round | Success rate | pass@3 | pass@1 | 每題 Pareto 候選數（平均） | PDAP / baseline |")
        out.append("|---|---:|---:|---:|---:|---:|---|")
        for group, arm in GROUPS:
            if group not in g:
                continue
            x = g[group]
            rounds = 5 if group == "multi-round" else 1
            ratio, n, *_ = pdap_comparison(data["combined"], group)
            value = "N/A" if ratio is None else f"{ratio:.3f}× (n={n})"
            out.append(f"| {group} (`{arm}`) | {rounds} | {pct(x['syntax_rate'])} | "
                       f"{pct(x['pass_at_3'])} | {pct(x['pass_at_1'])} | "
                       f"{x['mean_candidates']:.2f} | {value} |")
            sb = shell.get(model, {}).get("groups", {}).get("baseline")
            if group == "baseline" and sb:
                ratio, n = shell_ratio(data["combined"], shell[model]["combined"])
                value = "N/A" if ratio is None else f"{ratio:.3f}× (n={n})"
                out.append(f"| baseline + shell (`notools` + Bash) | 1 | {pct(sb['syntax_rate'])} | "
                           f"{pct(sb['pass_at_3'])} | {pct(sb['pass_at_1'])} | "
                           f"{sb['mean_candidates']:.2f} | {value} |")
        out.append("")
        out.append("pass@3: at least one correct final answer in three independent attempts per problem; N/A until all expected triplets are available.\n")
        out.append("PDAP / baseline: geometric mean of same-problem best-PDAP ratios over valid candidates from all attempts and rounds; lower is better. "
                   "n counts problems with valid candidates in both groups; comparison sets may differ. Raw per-problem PDAP and Top 3 remain in `summary.json`.\n")
        out.append("| 實驗組 | Attempts | API-error attempts | Any-round pass | Mean rounds run | "
                   "Attempts calling IC tools | Cost (USD) |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for group, arm in GROUPS:
            if group in g:
                x = g[group]
                out.append(f"| {group} | {x['attempts']} | {x['api_error_attempts']} | {pct(x['any_round_pass'])} | "
                           f"{x['mean_rounds']:.2f} | {x['tool_use_attempts']} | {x['cost_usd']:.2f} |")
        out.append("")
        out.append("IC tool calls: " + "; ".join(
            f"{group}: " + ", ".join(f"{k}={v}" for k, v in sorted(g[group]["tool_calls"].items()))
            for group, _ in GROUPS if group in g) + "\n")
        comb = data["combined"]
        contrib = {grp: 0 for grp, _ in GROUPS}
        present = {grp: 0 for grp, _ in GROUPS}
        wins = {grp: 0 for grp, _ in GROUPS}
        for p, c in comb.items():
            for grp in set(c["frontier_groups"]):
                present[grp] += 1
            for grp in c["frontier_groups"]:
                contrib[grp] += 1
            best = {k: v for k, v in c["best_pdap"].items() if v is not None}
            if best:
                m = min(best.values())
                for k, v in best.items():
                    wins[k] += v == m
        out.append("Combined frontier (all three groups pooled per problem):\n")
        out.append("| 實驗組 | Frontier points contributed | Problems where it reaches the combined frontier | "
                   "Problems where it holds the lowest PDAP (ties counted for each) |")
        out.append("|---|---:|---:|---:|")
        for grp, _ in GROUPS:
            out.append(f"| {grp} | {contrib[grp]} | {present[grp]} | {wins[grp]} |")
        out.append("")
    (DATA / "tables.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
