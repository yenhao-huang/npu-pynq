import math
import json
import os
import subprocess
from .ports import ports_of, CLK_RE

def measure_ppa(problem, sol, work, config):
    top = problem.top
    p = work / "ppa"
    p.mkdir(parents=True, exist_ok=True)
    ports = ports_of(sol.read_text(), top, p / "ports")
    clks = [n for n, d, w in ports if d == "input" and w == 1 and CLK_RE.search(n)]
    sdc = p / "grade.sdc"
    lines = [f"create_clock -name {c} -period {config.period_ns} [get_ports {c}]" for c in clks[:1]]
    if not clks:
        lines.append(f"create_clock -name vclk -period {config.period_ns}")
    clk = clks[0] if clks else "vclk"
    others = [n for n, d, w in ports if d == "input" and n not in clks[:1]]
    if others:
        lines.append(f"set_input_delay 0 -clock {clk} [get_ports {{{' '.join(others)}}}]")
    lines.append(f"set_output_delay 0 -clock {clk} [all_outputs]")
    sdc.write_text("\n".join(lines) + "\n")
    env = dict(os.environ, **config.tool_env, IC_ROOT=str(p / ".ic"))
    r = subprocess.run([str(config.venv / "ic"), "--compact", "ppa", "--files", str(sol), "--top", top,
                        "--sdc", str(sdc), "--clock-period-ns", str(config.period_ns),
                        "--input-activity", str(config.activity)],
                       capture_output=True, text=True, env=env, timeout=900)
    try:
        d = json.loads(r.stdout)
    except Exception:
        return dict(ppa_error=(r.stdout + r.stderr)[-300:])
    area = d.get("area", {}).get("cell_area_um2")
    t = d.get("timing", {})
    fmax = t.get("fmax_mhz")
    res = dict(area_um2=area, fmax_mhz=fmax, wns_ns=t.get("wns_ns"),
               power_w=d.get("power", {}).get("total_w"), ppa_ok=(r.returncode == 0 and d.get("ok") is True))
    if isinstance(fmax, (int, float)) and not isinstance(fmax, bool) and math.isfinite(fmax) and fmax > 0:
        res["delay_ns"] = 1000.0 / fmax
    return res
