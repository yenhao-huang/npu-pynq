import re
import shutil
import subprocess
from ..problems.patches import TB_PATCHES

def grade_shipped(problem, sol, work):
    g = work / "grade"
    shutil.rmtree(g, ignore_errors=True)
    g.mkdir(parents=True)
    shutil.copy(sol, g / "solution.v")
    for f in problem.directory.iterdir():  # testbench + stimulus files, from the pinned checkout
        if f.is_file() and not f.name.startswith("verified") and f.name not in ("design_description.txt", "makefile"):
            shutil.copy(f, g / f.name)
    for old, new in TB_PATCHES.get(problem.id, []):
        tb = (g / "testbench.v").read_text()
        assert old in tb, f"stale testbench patch for {problem.id}"
        (g / "testbench.v").write_text(tb.replace(old, new))
    r = subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "solution.v", "testbench.v"],
                       cwd=g, capture_output=True, text=True)
    if r.returncode:
        return False, "compile: " + r.stderr[-400:]
    try:
        r = subprocess.run(["vvp", "-n", "sim.vvp"], cwd=g, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "sim timeout"
    out = r.stdout
    ok = bool(re.search(r"Passed", out, re.I)) and not re.search(r"\bError\b|\bFailed\b|failures", out)
    return ok, out[-300:]
