import json
import re
import subprocess

CLK_RE = re.compile(r"clk|clock", re.I)
RST_RE = re.compile(r"rst|reset", re.I)

def ports_of(src, top, work):
    work.mkdir(parents=True, exist_ok=True)
    (work / "p.v").write_text(src)
    r = subprocess.run(["yosys", "-q", "-p", f"read_verilog -sv -lib p.v; write_json p.json"],
                       cwd=work, capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError("yosys ports: " + r.stderr[-300:])
    mod = json.loads((work / "p.json").read_text())["modules"][top]
    return [(n, p["direction"], len(p["bits"])) for n, p in mod["ports"].items()]
