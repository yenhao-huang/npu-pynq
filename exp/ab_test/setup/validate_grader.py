#!/usr/bin/env python3
"""Check the grader on a fresh machine: every reference must grade as expected.

Runs the RTLLM reference of each problem through the same three stages an
agent's solution goes through (shipped testbench, held-out differential test,
ppa) and compares with the values recorded on 2026-09-29. A mismatch means
the environment differs and results will not be comparable.
"""
import json, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "ab2"))
import harness as h  # noqa: E402

# radix2_div: the reference is wrong (fails its own testbench); held-out is skipped for it.
EXPECT_SHIPPED_FAIL = {"radix2_div"}
# References that are not synthesizable, or have no timing path, so have no ADP.
NO_ADP = {"clkgenerator", "float_multi", "synchronizer", "ROM", "multi_booth_8bit", "radix2_div"}
recorded = json.loads((EXP / "ab2" / "reference_ppa.json").read_text())


def check(pid):
    w = Path(tempfile.mkdtemp(prefix="ref_"))
    sol = w / "solution.v"
    sol.write_text(h.reference_source(pid))
    ok_s, _ = h.grade_shipped(pid, sol, w)
    ok_h, _ = h.grade_heldout(pid, sol, w)
    ppa = h.measure_ppa(pid, sol, w) if ok_s else {}
    problems = []
    if ok_s != (pid not in EXPECT_SHIPPED_FAIL):
        problems.append(f"shipped={ok_s}")
    if ok_h is False:
        problems.append("held-out failed on the reference")
    want = (recorded.get(pid) or {}).get("adp")
    if pid not in NO_ADP and want and ppa.get("adp") and abs(ppa["adp"] / want - 1) > 0.001:
        problems.append(f"adp {ppa['adp']} != recorded {want}")
    if pid not in NO_ADP and not ppa.get("adp"):
        problems.append("no adp")
    return pid, ppa.get("adp"), problems


with ThreadPoolExecutor(8) as ex:
    results = list(ex.map(check, sorted(h.PROBLEMS)))
bad = [r for r in results if r[2]]
for pid, adp, problems in results:
    print(f"{pid:24} adp={adp}  {'OK' if not problems else 'MISMATCH: ' + '; '.join(problems)}")
print(f"\n{len(results)} problems, {len(bad)} mismatches")
sys.exit(1 if bad else 0)
