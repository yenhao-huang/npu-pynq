#!/usr/bin/env python3
"""Independent correctness audit of an optimisation run.

For every attempt that was correct after round 1 and at the end, and whose
design changed, check the final design against its round-1 version with
`ic equiv` at a much higher effort than the agent used (20 seeds x 5000
cycles for sequential designs; a SAT proof for combinational ones).

usage: equiv_audit.py <experiment dir>   e.g. exp/ab_test/ab_multiround_ppa
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "ab2"))
import harness as h  # noqa: E402

exp = Path(sys.argv[1]).resolve()
env = dict(os.environ, **h.TOOL_ENV, IC_ROOT="/tmp/equiv_audit")
jobs = []
for f in glob.glob(str(exp / "results" / "*.json")):
    r = json.load(open(f))
    if not r["passed"] or not r.get("per_round") or not r["per_round"][0]["passed"]:
        continue
    key = f"{r['problem']}__{r['arm']}__{r['rep']}"
    first, last = exp / "results" / f"{key}.round1.v", exp / "results" / f"{key}.round{r['rounds']}.v"
    if first.read_text() != last.read_text():
        jobs.append((key, first, last, h.PROBLEMS[r["problem"]]["top"]))


def check(job):
    key, ref, dut, top = job
    p = subprocess.run([str(h.VENV / "ic"), "--compact", "equiv", "--ref-files", str(ref), "--dut-files", str(dut),
                        "--top", top, "--cycles", "5000", "--seeds", "20"], capture_output=True, text=True, env=env)
    try:
        d = json.loads(p.stdout)
        return key, d.get("equivalent"), d.get("method")
    except ValueError:
        return key, None, (p.stdout + p.stderr)[-200:]


with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(check, jobs))
ok = sum(1 for r in res if r[1] is True)
methods = {m: sum(1 for r in res if r[2] == m) for m in {r[2] for r in res}}
print(f"{len(res)} changed designs, {ok} equivalent to round 1, by method {methods}")
for r in res:
    if r[1] is not True:
        print("  NOT EQUIVALENT / UNKNOWN:", r)
sys.exit(0 if ok == len(res) else 1)
