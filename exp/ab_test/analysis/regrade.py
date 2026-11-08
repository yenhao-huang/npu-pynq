"""Re-grade one saved snapshot with the harness grader (correctness + PPA).

Run from exp/ab_test/:
    python3 analysis/regrade.py <problem_id> <path/to/roundN.v>
Prints the grade JSON plus PDAP = power_w * delay_ns * area_um2.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config  # noqa: E402
from src.problems.catalog import discover  # noqa: E402
from src.grade.evaluator import Grader  # noqa: E402


def main(problem_id, snapshot):
    config = Config()
    problem = discover(config.problems)[problem_id]
    with tempfile.TemporaryDirectory(prefix="regrade-") as d:
        sol = Path(d) / "solution.v"
        sol.write_text(Path(snapshot).read_text())
        grade = Grader(config).evaluate(problem, sol, Path(d) / "grade")
    if all(isinstance(grade.get(k), (int, float)) for k in ("power_w", "delay_ns", "area_um2")):
        grade["pdap"] = grade["power_w"] * grade["delay_ns"] * grade["area_um2"]
    print(json.dumps(grade, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
