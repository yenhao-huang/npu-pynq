"""Print "<arm> <reps> <problem ...>" lines: single-round (problem, rep) slots of an arm that still
lack a clean attempt, grouped by the --repetitions needed to regenerate the highest missing rep."""
import sys
from analyze import load, ROOT

model = sys.argv[1]
problems = sorted(p.parent.name for p in (ROOT.parent / "problems").rglob("design_description.txt"))
for group, arm in (("baseline", "notools"), ("single-round", "tools")):
    _, rows = load(model, group)
    clean = {(r["problem"], r["rep"]) for r in rows if not r["_api_error"]}
    need = {}
    for p in problems:
        missing = [rep for rep in range(3) if (p, rep) not in clean]
        if missing:
            need.setdefault(max(missing) + 1, []).append(p)
    for reps in sorted(need):
        print(arm, reps, " ".join(need[reps]))
