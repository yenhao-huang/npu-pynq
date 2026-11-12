"""Rerun the CUSTOM (A) versus CVA6 (B) scalar core comparison.

    python exp/scalar_core_compare/run.py                  # build, simulate, print the tables
    python exp/scalar_core_compare/run.py --update-readme  # also rewrite the results block of README.md
    python exp/scalar_core_compare/run.py --no-sim         # only parse the logs of the last run

Steps (from the repository root):
  1. `make -C src/test sim-cva6 sim-cva6-custom`: tb_scalar_core_cva6 under
     Verilator, once with SCALAR_IMPL="CVA6" on the RISC-V translations of the
     ISA test programs (src/test/vectors/risc_v_npu/isa_to_rv.py) and once with
     SCALAR_IMPL="CUSTOM" on the original ISA programs. Same bench, same memory
     model and LFSR, same simulator, same golden images.
  2. `make -C src/test sim TESTS=scalar_core`: the existing iverilog bench of A,
     as a cross-check of A's cycle counts.
  3. Parse src/test/build/scalar_core_cva6_{cva6,custom}.log and
     src/test/build/scalar_core.log, and size the programs (ISA words x 8 bytes
     versus RISC-V binary bytes).

On a Windows host make is MSYS2's (PATH gets /c/msys64/usr/bin and
/c/msys64/ucrt64/bin in front) and the C++ build runs in WSL; see the comment
above sim-cva6 in src/test/Makefile.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.test.vectors.risc_v_npu import isa_to_rv  # noqa: E402

TEST = ROOT / "src" / "test"
BUILD = TEST / "build"
README = Path(__file__).with_name("README.md")
BEGIN, END = "<!-- results:begin (written by run.py) -->", "<!-- results:end -->"

CASE = re.compile(
    r"^case (\S+): (\d+) jobs, (\d+) cycles, first_job (-?\d+), job_gap (\d+)/(\d+)/(\d+), "
    r"tail (-?\d+), \d+ reads, \d+ writes, prog (\d+) words (PASS|MISMATCH)$")
CVA6 = re.compile(
    r"^cva6 (\S+): first_fetch (-?\d+), fetches (\d+), mmio_wr (\d+), mmio_rd (\d+), mmio_wr_lat ([\d.]+)$")
IVERILOG = re.compile(r"^case (\S+): (\d+) jobs, (\d+) cycles, \d+ reads, \d+ writes$")


def make(*targets: str) -> None:
    env = dict(os.environ)
    cmd = ["make", "-C", str(TEST), *targets]
    if platform.system() == "Windows":
        env["PATH"] = r"C:\msys64\usr\bin;C:\msys64\ucrt64\bin;" + env["PATH"]
        cmd[0] = r"C:\msys64\usr\bin\make.exe"
        cmd[2] = "src/test"
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def parse_bench(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if m := CASE.match(line):
            name = m[1]
            rows[name] = dict(jobs=int(m[2]), cycles=int(m[3]), first_job=int(m[4]),
                              gap_min=int(m[5]), gap_avg=int(m[6]), gap_max=int(m[7]),
                              tail=int(m[8]), prog_words=int(m[9]), ok=m[10] == "PASS")
        elif m := CVA6.match(line):
            rows[m[1]].update(first_fetch=int(m[2]), fetches=int(m[3]), mmio_wr=int(m[4]),
                              mmio_rd=int(m[5]), mmio_wr_lat=float(m[6]))
    if not rows:
        raise SystemExit(f"no case lines in {path}")
    return rows


def parse_iverilog(path: Path) -> dict[str, int]:
    out = {}
    for line in path.read_text().splitlines():
        if m := IVERILOG.match(line):
            out[m[1]] = int(m[3])
    return out


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render(a: dict, b: dict, iv: dict, sizes: dict) -> str:
    dirty = git("status", "--porcelain", "--", "src", "exp")
    out = [f"Commit `{git('rev-parse', '--short', 'HEAD')}` "
           f"({'with uncommitted changes under src/ or exp/' if dirty else 'clean'}), "
           f"branch `{git('rev-parse', '--abbrev-ref', 'HEAD')}`.", ""]
    out += ["### Correctness and cycles", "",
            "| Case | Jobs | A cycles | B cycles | B / A | B - A | A image | B image | A cycles (iverilog) |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for name, ra in a.items():
        rb = b[name]
        jobs = ra["jobs"] if ra["jobs"] == rb["jobs"] else f"{ra['jobs']} / {rb['jobs']}"
        out.append(f"| {name} | {jobs} | {ra['cycles']} | {rb['cycles']} | "
                   f"{rb['cycles'] / ra['cycles']:.2f} | {rb['cycles'] - ra['cycles']} | "
                   f"{'PASS' if ra['ok'] else 'FAIL'} | {'PASS' if rb['ok'] else 'FAIL'} | "
                   f"{iv.get(name, 'n/a')} |")
    out += ["", "### Where the cycles go", "",
            "Clock cycles from ISA_CONTROL.START to the first matrix-core START (first job), "
            "between consecutive STARTs (min / avg / max), and from the last START until "
            "busy falls (tail).", "",
            "| Case | A first job | B first job | A job gap | B job gap | A tail | B tail |",
            "|---|---:|---:|---|---|---:|---:|"]
    for name, ra in a.items():
        rb = b[name]
        gap = lambda r: f"{r['gap_min']} / {r['gap_avg']} / {r['gap_max']}" if r["jobs"] > 1 else "-"
        val = lambda v: "-" if v < 0 else str(v)
        out.append(f"| {name} | {val(ra['first_job'])} | {val(rb['first_job'])} | {gap(ra)} | "
                   f"{gap(rb)} | {val(ra['tail'])} | {val(rb['tail'])} |")
    out += ["", "CVA6 side only: cycles from START to CVA6's first DDR read (reset release and "
            "I$ invalidation), CVA6 DDR reads (I$ line fills of 16 bytes; the programs read and "
            "write no DDR data), MMIO writes and reads, and the average cycles from an MMIO AW "
            "handshake to its B handshake (DOORBELL is held while the job queue is full).", "",
            "| Case | First fetch | I$ fills | MMIO writes | MMIO reads | MMIO write latency |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, rb in b.items():
        out.append(f"| {name} | {rb['first_fetch']} | {rb['fetches']} | {rb['mmio_wr']} | "
                   f"{rb['mmio_rd']} | {rb['mmio_wr_lat']:.1f} |")
    out += ["", "### Program size", "",
            "| Case | A: ISA words | A bytes | B: RV32I instructions | B bytes | B / A |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, (isa_words, rv_bytes) in sizes.items():
        out.append(f"| {name} | {isa_words} | {isa_words * 8} | {rv_bytes // 4} | {rv_bytes} | "
                   f"{rv_bytes / (isa_words * 8):.2f} |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-sim", action="store_true", help="parse the logs of the last run only")
    parser.add_argument("--update-readme", action="store_true", help="rewrite the results block of README.md")
    args = parser.parse_args()
    if not args.no_sim:
        make("sim-cva6", "sim-cva6-custom")
        (BUILD / "scalar_core.ok").unlink(missing_ok=True)
        make("sim", "TESTS=scalar_core")
    a = parse_bench(BUILD / "scalar_core_cva6_custom.log")
    b = parse_bench(BUILD / "scalar_core_cva6_cva6.log")
    iv = parse_iverilog(BUILD / "scalar_core.log")
    sizes = {isa_case.name: (isa_case.prog_len, rv_bytes) for isa_case, _, rv_bytes in isa_to_rv.build()}
    text = render(a, b, iv, sizes)
    print(text)
    failed = [n for n in a if not (a[n]["ok"] and b[n]["ok"])]
    if args.update_readme:
        readme = README.read_text()
        head, rest = readme.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        README.write_text(f"{head}{BEGIN}\n{text}\n{END}{tail}")
        print(f"updated {README}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
