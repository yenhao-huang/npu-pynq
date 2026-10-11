"""Re-grade every frontier Top 3 snapshot in evidence/top3.json and compare PPA with the run.

Run from exp/ab_test/:  python3 analysis/verify_top3.py [model ...]
Writes evidence/top3_verify_<model>.json; prints mismatches.
"""
import json
import os
import math
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AB_RESULTS", ROOT.parent / "results")).resolve()  # local run archive by default
sys.path.insert(0, str(ROOT.parent))
from src.config import Config  # noqa: E402
from src.problems.catalog import discover  # noqa: E402
from src.grade.evaluator import Grader  # noqa: E402

KEYS = ("area_um2", "power_w", "delay_ns")


def regrade(problem, path, config, grader):
    with tempfile.TemporaryDirectory(prefix="verify-") as d:
        sol = Path(d) / "solution.v"
        sol.write_text((DATA / path).read_text())
        return grader.evaluate(problem, sol, Path(d) / "grade")


def main(models):
    config = Config()
    problems, grader = discover(config.problems), Grader(config)
    top3 = json.loads((DATA / "top3.json").read_text())
    for model in models or list(top3):
        entries = [(group, e) for group, by_problem in top3[model].items()
                   for es in by_problem.values() for e in es]
        def check(item):
            group, e = item
            g = regrade(problems[e["problem"]], e["solution_path"], config, grader)
            same = g.get("passed") is True and all(
                isinstance(g.get(k), (int, float)) and math.isclose(g[k], e[k], rel_tol=1e-9) for k in KEYS)
            return dict(group=group, problem=e["problem"], solution_path=e["solution_path"],
                        expected={k: e[k] for k in KEYS}, regraded={k: g.get(k) for k in KEYS},
                        passed=g.get("passed"), match=same)
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(check, entries))
        (DATA / f"top3_verify_{model}.json").write_text(json.dumps(rows, indent=1))
        bad = [r for r in rows if not r["match"]]
        print(f"{model}: {len(rows) - len(bad)}/{len(rows)} reproduced exactly")
        for r in bad:
            print("  MISMATCH", r["group"], r["problem"], r["solution_path"], r["expected"], r["regraded"])


if __name__ == "__main__":
    main(sys.argv[1:])
