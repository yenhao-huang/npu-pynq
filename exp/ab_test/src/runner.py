"""Compose problem, agent, grading, and logging components for one attempt."""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
from .agent.context import Context
from .agent.rounds import run_rounds
from .agent.parse import summarize
from .logging.recorder import Recorder
from .grade.pareto import frontier, top3_pdap, OBJECTIVES

def attempt(problem, arm, rep, config, model, grader, output):
    recorder = Recorder(output / f"{problem.id}__{arm}__{rep}")
    recorder.json("config.json", asdict(config))
    row = dict(problem=problem.id, category=problem.category, arm=arm, rep=rep,
               requested_rounds=config.rounds, passed=False, per_round=[])
    context = Context(problem.description(), arm, config.rounds)
    try:
        with TemporaryDirectory(prefix="ab-rounds-") as directory:
            work = Path(directory)
            records = run_rounds(context, model, work, recorder)
            row.update(summarize([record["response"].stdout for record in records]))
            row.update(rounds=len(records), wall_s=sum(r["response"].wall_s for r in records),
                       timed_out=any(r["response"].timed_out for r in records))
            # Evaluate after the conversation so hidden grading cannot affect later rounds.
            for record in records:
                number = record["round"]
                grade = dict(round=number, passed=False)
                if record["solution"] is not None:
                    grade["solution_path"] = str((recorder.directory / f"round{number}.v").relative_to(output))
                if record["error"]:
                    grade["failure"] = record["error"]
                elif record["solution"] is None:
                    grade["failure"] = "no solution produced"
                else:
                    solution = work / f"round{number}.v"
                    solution.write_text(record["solution"])
                    try:
                        grade.update(grader.evaluate(problem, solution, work / f"grade{number}"))
                    except Exception as exc:
                        grade.update(passed=False, failure=f"grading: {exc!r}")
                recorder.json(f"round{number}.grade.json", grade)
                row["per_round"].append(grade)
            if row["per_round"]:
                row["final_round"] = row["per_round"][-1]
                if row["final_round"].get("failure"):
                    row["failure"] = row["final_round"]["failure"]
    except Exception as exc:
        row.update(passed=False, failure=f"harness: {exc!r}")
    row["pareto_frontier"] = frontier(row["per_round"])
    row["top3_pdap"] = top3_pdap(row["pareto_frontier"])
    row["passed"] = bool(row["pareto_frontier"])
    recorder.json("frontier.json", {"objectives": list(OBJECTIVES), "direction": "minimize",
                                   "candidates": row["pareto_frontier"],
                                   "top3_pdap": row["top3_pdap"]})
    recorder.json("result.json", row)
    print(problem.id, arm, rep, "PASS" if row["passed"] else "FAIL", flush=True)
    return row
