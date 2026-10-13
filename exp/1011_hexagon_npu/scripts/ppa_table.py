"""Render out-of-context PPA results (ppa_module.tcl summary.json) as Markdown.

Usage: python3 ppa_table.py <artifact dir> [name ...]

<artifact dir> is a downloaded exp-ppa-<id> artifact (one folder per run, each
holding summary.json). With names, only those runs are printed, in that order;
the first one is the "before" reference for the delta columns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
    ("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP"), ("bram_tiles", "BRAM"),
    ("carry4", "CARRY4"), ("wns_ns", "WNS (ns)"), ("fmax_mhz", "Fmax (MHz)"),
    ("power_dynamic_w", "Dyn. (W)"), ("power_total_w", "Total (W)"),
)


def load(root: Path) -> dict[str, dict]:
    runs = {}
    for summary in sorted(root.glob("*/summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        runs[data["name"]] = data
    return runs


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.1f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def table(runs: dict[str, dict], names: list[str]) -> str:
    head = "| Run | " + " | ".join(label for _, label in COLUMNS) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in COLUMNS) + " |"
    lines = [head, rule]
    for name in names:
        run = runs[name]
        lines.append(f"| `{name}` | " + " | ".join(fmt(run.get(key, "")) for key, _ in COLUMNS) + " |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    root = Path(argv[1])
    runs = load(root)
    names = argv[2:] or list(runs)
    print(table(runs, names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
