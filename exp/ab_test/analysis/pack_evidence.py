"""Pack selected clean attempts and Top 3 RTL snapshots from a full run.

Run from exp/ab_test/ after generating aggregate files in results/:
    python3 analysis/pack_evidence.py --source results --target evidence

The compact manifest preserves measured grades, usage, provenance, and each
snapshot's independently checked syntax verdict. Full CLI streams remain local.
"""
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

FIELDS = (
    "problem", "category", "arm", "rep", "requested_rounds", "per_round",
    "tool_calls", "usage", "cost_usd", "rounds", "wall_s", "timed_out",
    "failure", "final_round",
)


def pack(source, target, groups):
    analyze.DATA = source
    entries = []
    for model in analyze.MODELS:
        for group in groups:
            _, rows = analyze.load(model, group)
            for row in rows:
                count = max(row.get("rounds", 0), len(row["per_round"]), 1)
                syntax = {
                    str(i): bool(analyze.syntax_ok(row["_dir"] / f"round{i}.v"))
                    for i in range(1, count + 1)
                }
                entries.append({
                    "model": model,
                    "group": group,
                    "run": str(row["_run"].relative_to(source)),
                    "attempt_dir": str(row["_dir"].relative_to(source)),
                    "syntax_ok_rounds": syntax,
                    "record": {key: row[key] for key in FIELDS if key in row},
                })
    target.mkdir(parents=True, exist_ok=True)
    (target / "attempts.jsonl").write_text(
        "".join(json.dumps(entry, separators=(",", ":")) + "\n" for entry in entries)
    )
    return len(entries)


def copy_top3(source, target, paths):
    for relative in paths:
        src = source / relative
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results"))
    parser.add_argument("--target", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    primary = pack(source, target, analyze.GROUPS)
    shell = pack(source / "shell", target / "shell", ("baseline",))
    top3 = json.loads((source / "top3.json").read_text())
    primary_paths = {
        entry["solution_path"]
        for model in top3.values() for group in model.values()
        for rows in group.values() for entry in rows
    }
    copy_top3(source, target, primary_paths)
    shell_summary = json.loads((source / "shell" / "summary.json").read_text())
    shell_paths = {
        entry["solution_path"]
        for model in shell_summary["models"].values()
        for rows in model["groups"]["baseline"]["top3"].values()
        for entry in rows
    }
    copy_top3(source / "shell", target / "shell", shell_paths)
    analyze.DATA = source
    for row in analyze.load("claude-opus-5-5", "multi-round")[1]:
        if row["problem"] == "JC_counter":
            # The acceptance notes retain this clean alternative to the best round.
            copy_top3(source, target, {str((row["_dir"] / "round4.v").relative_to(source))})
            break
    for name in (
        "summary.json", "extras.json", "report_tables.md", "tables.md",
        "top3.json", "top3_combined.md", "multi_round_analysis.json",
        "top3_verify_claude-opus-5-5.json",
        "top3_verify_claude-sonnet-5-5.json", "top3_verify_gpt-6-sol.json",
    ):
        shutil.copy2(source / name, target / name)
    for name in ("top3_verify_baseline.json",):
        shutil.copy2(source / "shell" / name, target / "shell" / name)
    analyze.DATA = target / "shell"
    with redirect_stdout(StringIO()):
        analyze.main()  # Only the selected shell baseline is present in this manifest.
    print(f"Packed {primary} primary and {shell} shell attempts; "
          f"kept {len(primary_paths) + len(shell_paths)} Top 3 snapshots")


if __name__ == "__main__":
    main()
