"""Run with: python -m src.main [problem ...] --rounds N."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import os
import random
import uuid
from .config import Config
from .problems.catalog import discover
from .agent.llm import ClaudeModel
from .agent.codex import CodexModel
from .grade.evaluator import Grader
from .runner import attempt
from .grade.pareto import grouped_frontiers, OBJECTIVES
from .logging.recorder import Recorder

def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value

def main(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="*")
    parser.add_argument("--problems-dir", type=Path, default=defaults.problems)
    parser.add_argument("--repo", type=Path, default=defaults.repo)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--rounds", type=positive, default=os.getenv("AB_ROUNDS", "1"))
    parser.add_argument("--repetitions", "--reps", dest="repetitions", type=positive,
                        default=os.getenv("AB_REPS", "3"), help="independent attempts per problem and arm")
    parser.add_argument("--parallel", type=positive, default=os.getenv("AB_PAR", "6"))
    parser.add_argument("--model", default=os.getenv("AB_MODEL", "sonnet"))
    parser.add_argument("--backend", choices=["auto", "claude", "codex"], default=os.getenv("AB_BACKEND", "auto"),
                        help="model CLI; auto picks codex for gpt-* models")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                        default=os.getenv("AB_EFFORT", "medium"))
    parser.add_argument("--budget", default=os.getenv("AB_BUDGET", "3.00"), help="USD per model invocation")
    parser.add_argument("--arms", nargs="+", choices=["notools", "tools", "tools_directed"],
                        default=os.getenv("AB_ARMS", "notools,tools,tools_directed").split(","))
    parser.add_argument("--shell", action="store_true", default=os.getenv("AB_SHELL") == "1",
                        help="give every arm a shell (Bash); arms then differ only by the ic tools")
    parser.add_argument("--list", action="store_true", help="list problems without executing models")
    args = parser.parse_args(argv)
    backend = args.backend
    if backend == "auto":
        backend = "codex" if args.model.startswith("gpt") else "claude"
    config = Config(problems=args.problems_dir.resolve(), repo=args.repo.resolve(),
                    output=args.output.resolve(), model=args.model, backend=backend, budget=args.budget, rounds=args.rounds,
                    effort=args.effort, repetitions=args.repetitions, shell=args.shell)
    problems = discover(config.problems)
    if not problems:
        parser.error(f"no problems found in {config.problems}")
    selected = list(dict.fromkeys(args.problems)) or sorted(problems)
    missing = set(selected) - problems.keys()
    if missing:
        parser.error(f"unknown problems: {sorted(missing)}")
    if args.list:
        print("\n".join(selected))
        return
    output = config.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    jobs = [(problems[p], a, r) for r in range(config.repetitions) for p in selected for a in dict.fromkeys(args.arms)]
    random.Random(74).shuffle(jobs)
    run = partial(attempt, config=config, model=(CodexModel if backend == "codex" else ClaudeModel)(config), grader=Grader(config), output=output)
    print(f"{len(jobs)} attempts, {config.rounds} rounds; output: {output}", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = list(pool.map(lambda job: run(*job), jobs))
    Recorder(output).json("frontier.json", {
        "objectives": list(OBJECTIVES), "direction": "minimize",
        "problems": grouped_frontiers(rows),
    })

if __name__ == "__main__":
    main()
