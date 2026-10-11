import re
import shutil
import subprocess
from ..problems.catalog import reference_source, prefixed_reference
from ..problems.patches import HELDOUT_SKIP
from .ports import ports_of, CLK_RE, RST_RE

def heldout_tb(problem, ports, seed, cycles=2000):
    top = problem.top
    ins = [p for p in ports if p[1] == "input"]
    outs = [p for p in ports if p[1] == "output"]
    clks = [p[0] for p in ins if p[2] == 1 and CLK_RE.search(p[0])]
    rsts = [p[0] for p in ins if p[2] == 1 and RST_RE.search(p[0]) and p[0] not in clks]
    data = [p for p in ins if p[0] not in clks and p[0] not in rsts]
    active_low = {r: bool(re.search(r"(_n|n)$", r, re.I)) for r in rsts}

    L = ["`timescale 1ns/1ps", "module tb_heldout;"]
    for n, _, w in ins:
        L.append(f"  reg [{w-1}:0] {n};")
    for n, _, w in outs:
        L.append(f"  wire [{w-1}:0] dut_{n}, ref_{n};")
    conn = lambda pre: ", ".join([f".{n}({n})" for n, _, _ in ins] + [f".{n}({pre}_{n})" for n, _, _ in outs])
    L.append(f"  {top} dut({conn('dut')});")
    L.append(f"  ref_{top} refm({conn('ref')});")
    L.append("  integer errors = 0, cyc = 0, seed = %d;" % seed)
    for i, c in enumerate(clks):
        half = 5 + 2 * i
        L.append(f"  initial begin {c} = 0; forever #{half} {c} = ~{c}; end")

    def assign_rst(active):
        return " ".join(f"{r} = {int(active != active_low[r])};" for r in rsts)

    def rand_data():
        s = []
        for n, _, w in data:
            words = (w + 31) // 32
            s.append(f"{n} = {{" + ", ".join(["$random(seed)"] * words) + "};")
        return " ".join(s)

    def compare():
        s = []
        for n, _, w in outs:
            s.append(f"if (^ref_{n} !== 1'bx && dut_{n} !== ref_{n}) begin errors = errors + 1; "
                     f"if (errors <= 5) $display(\"MISMATCH cyc=%0d {n} ref=%h dut=%h\", cyc, ref_{n}, dut_{n}); end")
        return " ".join(s)

    L.append("  initial begin")
    L.append(f"    {rand_data()} {assign_rst(True)}")
    if clks:
        main = clks[0]
        L.append(f"    repeat (3) @(negedge {main});")
        L.append(f"    {assign_rst(False)}")
        L.append(f"    for (cyc = 0; cyc < {cycles}; cyc = cyc + 1) begin")
        L.append(f"      @(negedge {main}); {compare()}")
        L.append(f"      {rand_data()}")
        if rsts:
            L.append(f"      if (cyc == {cycles // 2}) begin {assign_rst(True)} end")
            L.append(f"      if (cyc == {cycles // 2 + 2}) begin {assign_rst(False)} end")
        L.append("    end")
    else:
        L.append(f"    #1 {assign_rst(False)}")
        L.append(f"    for (cyc = 0; cyc < {cycles}; cyc = cyc + 1) begin")
        L.append(f"      {rand_data()} #5; {compare()} #5;")
        L.append("    end")
    L.append('    if (errors == 0) $display("HELDOUT PASS"); else $display("HELDOUT FAIL errors=%0d", errors);')
    L.append("    $finish;")
    L.append("  end")
    L.append("endmodule")
    return "\n".join(L) + "\n"

def grade_heldout(problem, sol, work):
    if problem.id in HELDOUT_SKIP:
        return None, "held-out skipped: reference is wrong"
    h = work / "heldout"
    shutil.rmtree(h, ignore_errors=True)
    h.mkdir(parents=True)
    top = problem.top
    try:
        ports = ports_of(reference_source(problem), top, h / "ports")
    except Exception as e:
        return None, f"port extraction failed: {e}"
    (h / "ref.v").write_text(prefixed_reference(problem))
    shutil.copy(sol, h / "solution.v")
    seed = int.from_bytes(problem.id.encode(), "little") % 100000 + 1
    (h / "tb.v").write_text(heldout_tb(problem, ports, seed))
    r = subprocess.run(["iverilog", "-g2012", "-o", "h.vvp", "solution.v", "ref.v", "tb.v"],
                       cwd=h, capture_output=True, text=True)
    if r.returncode:
        return False, "compile: " + r.stderr[-400:]
    try:
        r = subprocess.run(["vvp", "-n", "h.vvp"], cwd=h, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "heldout timeout"
    return ("HELDOUT PASS" in r.stdout), r.stdout[-400:]
