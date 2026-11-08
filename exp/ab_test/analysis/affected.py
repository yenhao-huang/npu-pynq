"""Print problems that still lack any clean multi-round attempt (one per problem is enough)."""
import sys
from analyze import load, ROOT

model, group = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "multi-round"
problems = sorted(p.parent.name for p in (ROOT.parent / "problems").rglob("design_description.txt"))
_, rows = load(model, group)
clean = {r["problem"] for r in rows if not r["_api_error"]}
print(" ".join(p for p in problems if p not in clean))
