"""Render the comparison table in the format of docs/goals/ab_test_human.md from summary.json.

Run from exp/ab_test/:  python3 analysis/analyze.py && python3 analysis/report_tables.py
Writes evidence/report_tables.md (main table per model) and evidence/top3.json (per model and
group, the frontier Top 3 by PDAP for every problem, with snapshot paths for reproduction).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AB_RESULTS", ROOT.parent / "results")).resolve()  # local run archive by default
GROUPS = [("baseline", "notools", 1), ("single-round", "tools", 1), ("multi-round", "tools", 5)]


def pct(x):
    return "N/A" if x is None else f"{100 * x:.1f}%"


def main():
    s = json.loads((DATA / "summary.json").read_text())
    shell_path = DATA / "shell" / "summary.json"
    shell = json.loads(shell_path.read_text())["models"] if shell_path.exists() else {}
    out, top3 = [], {}
    for model, data in s["models"].items():
        g = data["groups"]
        if not g:
            continue
        n = next(iter(g.values()))["problems"]
        budget = "無（Codex CLI 無預算參數）" if model.startswith("gpt") else "`$3.00`"
        out.append(f"### `{model}`\n")
        out.append(f"**實驗設定**：模型 `{model}`｜effort `medium`｜題數 `{n}`｜每題獨立重複 "
                   f"`3` 次（multi-round `1` 次）｜每輪預算 {budget}\n")
        out.append("| 實驗組 | Round | Success rate | FUNCTION | pass@1 | 每題 Pareto 候選數（平均） | Frontier Top 3：PDAP |")
        out.append("|---|---:|---:|---:|---:|---:|---|")
        for group, arm, rounds in GROUPS:
            if group not in g:
                out.append(f"| {group} (`{arm}`) | {rounds} | — | — | — | — | 未完成 |")
                continue
            x = g[group]
            cell = "<br>".join(
                f"{p}：" + (" / ".join(f"{c['pdap']:.3g}" for c in x["top3"][p]) or "N/A")
                for p in sorted(x["top3"]))
            out.append(f"| {group} (`{arm}`) | {rounds} | {pct(x['syntax_rate'])} | "
                       f"{x['function_problems']} / {x['problems']} | {pct(x['pass_at_1'])} | "
                       f"{x['mean_candidates']:.2f} | <details><summary>{x['problems']} 題</summary>{cell}</details> |")
            top3.setdefault(model, {})[group] = x["top3"]
            sb = shell.get(model, {}).get("groups", {}).get("baseline")
            if group == "baseline" and sb:
                cell = "<br>".join(
                    f"{p}：" + (" / ".join(f"{c['pdap']:.3g}" for c in sb["top3"][p]) or "N/A")
                    for p in sorted(sb["top3"]))
                out.append(f"| baseline + shell (`notools` + Bash) | 1 | {pct(sb['syntax_rate'])} | "
                           f"{sb['function_problems']} / {sb['problems']} | {pct(sb['pass_at_1'])} | "
                           f"{sb['mean_candidates']:.2f} | <details><summary>{sb['problems']} 題</summary>{cell}</details> |")
        out.append("")
    (DATA / "report_tables.md").write_text("\n".join(out))
    (DATA / "top3.json").write_text(json.dumps(top3, indent=1))
    print("\n".join(line[:220] for line in out))


if __name__ == "__main__":
    main()
