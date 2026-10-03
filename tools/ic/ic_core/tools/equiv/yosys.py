"""Yosys + Icarus equivalence backend.

    ref, dut --yosys -lib--> ports (must match)
             --yosys proc--> any flip-flops?
       no  : miter -equiv + sat -prove          -> proof or counterexample
       yes : iverilog differential testbench    -> first mismatching cycle

Both designs are loaded under renamed modules (`ref_*`, `dut_*`) so that
submodules sharing a name cannot collide.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ...errors import InvalidInput
from ...process import run as run_process
from ...registry import backend
from . import EquivIn, EquivOut, Mismatch

CLK_RE = re.compile(r"clk|clock", re.I)
RST_RE = re.compile(r"rst|reset", re.I)
MISMATCH_RE = re.compile(r"@ic mismatch seed=(\d+) cyc=(\d+) port=(\S+) ref=(\S+) dut=(\S+)")
SAT_INPUT_RE = re.compile(r"^\s*\\?in_(\S+)\s+\S+\s+\S+\s+(\S+)\s*$", re.M)


def _read(files: list[str], cwd: Path) -> str:
    text = []
    for name in files:
        path = Path(name)
        if not path.is_absolute():
            path = cwd / path
        if not path.is_file():
            raise InvalidInput("equiv source does not exist", missing=[name])
        text.append(path.read_text())
    return "\n".join(text)


def _prefixed(src: str, prefix: str) -> str:
    """Rename every module the source defines, and every reference to it."""
    names = set(re.findall(r"^\s*module\s+(\w+)", src, re.M))
    for name in sorted(names, key=len, reverse=True):
        src = re.sub(rf"\b{re.escape(name)}\b", prefix + name, src)
    return src


def _ports(src: Path, top: str, work: Path, log: Path) -> list[tuple[str, str, int]]:
    out = work / f"{src.stem}.ports.json"
    run_process(["yosys", "-q", "-p", f"read_verilog -sv -lib {src}; write_json {out}"],
                log_path=log, cwd=work, timeout_s=120, append=True)
    if not out.exists():
        raise InvalidInput(f"could not read the ports of {top}")
    modules = json.loads(out.read_text())["modules"]
    if top not in modules:
        raise InvalidInput(f"module {top!r} not found", known=sorted(modules))
    return [(n, p["direction"], len(p["bits"])) for n, p in modules[top]["ports"].items()]


def _is_sequential(src: Path, top: str, work: Path, log: Path) -> bool:
    stat = work / f"{src.stem}.stat.json"
    run_process(["yosys", "-q", "-p",
                 f"read_verilog -sv {src}; hierarchy -top {top}; proc; flatten; "
                 f"tee -q -o {stat} stat -json"],
                log_path=log, cwd=work, timeout_s=300, append=True)
    if not stat.exists():
        return True  # cannot elaborate for synthesis: fall back to simulation
    cells = json.loads(stat.read_text()).get("design", {}).get("num_cells_by_type", {})
    return any(re.search(r"dff|dlatch|mem", kind, re.I) for kind in cells)


def _testbench(top: str, ports, clocks, resets, active_low, cycles: int, seed: int) -> str:
    ins = [p for p in ports if p[1] == "input"]
    outs = [p for p in ports if p[1] == "output"]
    data = [p for p in ins if p[0] not in clocks and p[0] not in resets]
    lines = ["`timescale 1ns/1ps", "module ic_equiv_tb;"]
    lines += [f"  reg [{w-1}:0] {n};" for n, _, w in ins]
    lines += [f"  wire [{w-1}:0] r_{n}, d_{n};" for n, _, w in outs]

    def conn(pre):
        return ", ".join([f".{n}({n})" for n, _, _ in ins] + [f".{n}({pre}_{n})" for n, _, _ in outs])

    lines += [f"  ref_{top} u_ref({conn('r')});", f"  dut_{top} u_dut({conn('d')});",
              f"  integer errors = 0, cyc = 0, seed = {seed};"]
    for i, clk in enumerate(clocks):
        lines.append(f"  initial begin {clk} = 0; forever #{5 + 2 * i} {clk} = ~{clk}; end")

    def rst(active):
        return " ".join(f"{r} = {int(active != active_low[r])};" for r in resets)

    rand = " ".join(f"{n} = {{{', '.join(['$random(seed)'] * ((w + 31) // 32))}}};" for n, _, w in data)
    check = " ".join(
        f"if (^r_{n} !== 1'bx && d_{n} !== r_{n}) begin errors = errors + 1; if (errors <= 5) "
        f"$display(\"@ic mismatch seed={seed} cyc=%0d port={n} ref=%h dut=%h\", cyc, r_{n}, d_{n}); end"
        for n, _, _ in outs)
    lines += ["  initial begin", f"    {rand} {rst(True)}"]
    if clocks:
        main = clocks[0]
        lines += [f"    repeat (3) @(negedge {main});", f"    {rst(False)}",
                  f"    for (cyc = 0; cyc < {cycles}; cyc = cyc + 1) begin",
                  f"      @(negedge {main}); {check}", f"      {rand}"]
        if resets:
            lines += [f"      if (cyc == {cycles // 2}) begin {rst(True)} end",
                      f"      if (cyc == {cycles // 2 + 2}) begin {rst(False)} end"]
        lines.append("    end")
    else:
        lines += [f"    #1 {rst(False)}", f"    for (cyc = 0; cyc < {cycles}; cyc = cyc + 1) begin",
                  f"      {rand} #5; {check} #5;", "    end"]
    lines += ['    $display("@ic errors=%0d", errors);', "    $finish;", "  end", "endmodule"]
    return "\n".join(lines) + "\n"


@backend("equiv", "yosys", requires="yosys", version_cmd=["yosys", "-V"])
class YosysEquiv:
    def equiv(self, params: EquivIn, ctx) -> EquivOut:
        started = time.monotonic()
        cwd = Path(ctx.cwd)
        work = ctx.run.work
        log = ctx.run.artifacts / "equiv.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("")
        ref_src, dut_src = _read(params.ref_files, cwd), _read(params.dut_files, cwd)
        ref_v, dut_v = work / "ref.v", work / "dut.v"
        ref_v.write_text(_prefixed(ref_src, "ref_"))
        dut_v.write_text(_prefixed(dut_src, "dut_"))
        top = params.top

        ref_ports = _ports(ref_v, "ref_" + top, work, log)
        dut_ports = _ports(dut_v, "dut_" + top, work, log)
        if sorted(ref_ports) != sorted(dut_ports):
            raise InvalidInput("ref and dut ports differ",
                               ref=sorted(ref_ports), dut=sorted(dut_ports))

        common = dict(backend="yosys", backend_version=ctx.backend_version,
                      report=ctx.run.handle("equiv.log"))
        sequential = (_is_sequential(ref_v, "ref_" + top, work, log)
                      or _is_sequential(dut_v, "dut_" + top, work, log))

        if not sequential:
            script = work / "miter.ys"
            script.write_text("\n".join([
                f"read_verilog -sv {ref_v}", f"read_verilog -sv {dut_v}",
                f"proc; flatten; opt_clean",
                f"miter -equiv -flatten -make_outputs -ignore_gold_x ref_{top} dut_{top} ic_miter",
                "hierarchy -top ic_miter",
                "sat -verify -prove trigger 0 -show-inputs -timeout 240 ic_miter",
            ]) + "\n")
            result = run_process(["yosys", "-s", str(script)], log_path=log, cwd=work,
                                 timeout_s=params.timeout_s, append=True)
            text = result.text()
            proved = "SUCCESS!" in text and result.exit_code == 0
            failed = "FAIL!" in text or "Proof failed" in text
            cex = {m.group(1): m.group(2) for m in SAT_INPUT_RE.finditer(text)} if failed else {}
            if proved or failed:
                return EquivOut(
                    ok=True, equivalent=proved, method="sat-proof", sequential=False,
                    counterexample=cex,
                    summary=["proved equivalent for every input" if proved
                             else "not equivalent: see counterexample"],
                    duration_s=round(time.monotonic() - started, 3), **common)
            # SAT could not decide (timeout or unsupported construct): simulate instead.

        ins = [p for p in ref_ports if p[1] == "input"]
        clocks = params.clock_ports or [n for n, _, w in ins if w == 1 and CLK_RE.search(n)]
        resets = params.reset_ports or [n for n, _, w in ins
                                         if w == 1 and RST_RE.search(n) and n not in clocks]
        active_low = {r: bool(re.search(r"n$", r, re.I)) for r in resets}
        mismatches, total, errors = [], 0, 0
        for seed in range(1, params.seeds + 1):
            tb = work / f"tb_{seed}.v"
            tb.write_text(_testbench(top, ref_ports, clocks, resets, active_low, params.cycles, seed * 7919))
            vvp = work / f"eq_{seed}.vvp"
            compiled = run_process(["iverilog", "-g2012", "-o", str(vvp), str(ref_v), str(dut_v), str(tb)],
                                   log_path=log, cwd=work, timeout_s=120, append=True)
            if compiled.exit_code != 0:
                return EquivOut(ok=False, method="random-sim", sequential=sequential,
                                summary=["compile failed"] + compiled.text().splitlines()[-5:],
                                duration_s=round(time.monotonic() - started, 3), **common)
            ran = run_process(["vvp", "-n", str(vvp)], log_path=log, cwd=work,
                              timeout_s=params.timeout_s, append=True)
            text = ran.text()
            found = re.search(r"@ic errors=(\d+)", text)
            if not found:
                return EquivOut(ok=False, method="random-sim", sequential=sequential,
                                summary=["simulation did not finish"],
                                duration_s=round(time.monotonic() - started, 3), **common)
            errors += int(found.group(1))
            total += params.cycles
            for m in MISMATCH_RE.finditer(text):
                if len(mismatches) < 5:
                    mismatches.append(Mismatch(seed=int(m.group(1)), cycle=int(m.group(2)),
                                               port=m.group(3), ref=m.group(4), dut=m.group(5)))
            if errors:
                break
        return EquivOut(
            ok=True, equivalent=errors == 0, method="random-sim", sequential=sequential,
            compared_cycles=total, mismatches=mismatches,
            summary=[f"{'no difference' if errors == 0 else f'{errors} mismatching output samples'} "
                     f"over {total} random cycles, clocks={clocks}, resets={resets}"],
            duration_s=round(time.monotonic() - started, 3),
            note="Random simulation is evidence, not proof; outputs the reference leaves X are not compared.",
            **common)
