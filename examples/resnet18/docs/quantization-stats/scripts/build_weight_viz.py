"""Render an interactive HTML page of the ResNet-18 weight distribution.

Two panels on one canvas:

- Panel A: global histogram of all folded weights, log-count, both before and
  after INT8 per-output-channel quantization overlaid.
- Panel B: per-layer density ridgeline. Each row uses its own x range from
  ``-max|W|`` to ``+max|W|`` so shallow-layer shape stays visible; original and
  dequantized densities are overlaid so the reconstruction quality is
  eyeballable. The right-hand columns state the layer's median quantization
  step and its relative RMS reconstruction error.

Usage (relative to the repository root):

    python examples/resnet18/docs/quantization-stats/scripts/build_weight_viz.py \\
        --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \\
        --output    examples/resnet18/model/weight_distribution.html

The output is self-contained (no external assets) and works offline. The
histogram data is inlined; the file is ~250 kB.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.export.torchvision_resnet18 import (
    _conv_specs,
    _folded_convolutions,
    load_checkpoint,
    quantize_conv_weight,
)


NBINS = 200
GLOBAL_XMIN, GLOBAL_XMAX = -4.0, 4.0
STAGE_COLORS = {
    "stem":   ("#cde2fb", "#86b6ef"),
    "layer1": ("#b7d3f6", "#5598e7"),
    "layer2": ("#9ec5f4", "#2a78d6"),
    "layer3": ("#86b6ef", "#1c5cab"),
    "layer4": ("#6da7ec", "#0d366b"),
}


def _stage_of(name: str) -> str:
    return "stem" if name.startswith("stem") else name.split(".")[0]


def _hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)[0]


def _compute(checkpoint: Path):
    state = load_checkpoint(checkpoint)
    folded = _folded_convolutions(state)
    order = [spec.command_id for spec in _conv_specs()]

    global_edges = np.linspace(GLOBAL_XMIN, GLOBAL_XMAX, NBINS + 1)
    global_orig = np.zeros(NBINS, dtype=np.int64)
    global_deq = np.zeros(NBINS, dtype=np.int64)
    total_weights = 0
    layers = []

    for name in order:
        weight = folded[name][0].astype(np.float64)
        qweight, scales = quantize_conv_weight(weight.astype(np.float32))
        q = np.transpose(qweight, (3, 2, 0, 1)).astype(np.int32)
        dequantized = q.astype(np.float64) * scales[:, None, None, None]

        max_abs = float(np.abs(weight).max())
        lo, hi = -max_abs * 1.05, max_abs * 1.05
        edges = np.linspace(lo, hi, NBINS + 1)
        orig_hist = _hist(weight.ravel(), edges)
        deq_hist = _hist(dequantized.ravel(), edges)
        error = weight - dequantized

        global_orig += _hist(weight.ravel(), global_edges)
        global_deq += _hist(dequantized.ravel(), global_edges)
        total_weights += weight.size

        layers.append({
            "name": name,
            "stage": _stage_of(name),
            "count": int(weight.size),
            "max_abs": max_abs,
            "edges": edges.tolist(),
            "orig_hist": orig_hist.tolist(),
            "deq_hist": deq_hist.tolist(),
            "channel_scale_median": float(np.median(scales)),
            "rms_error_relative": float(
                np.linalg.norm(error) / max(np.linalg.norm(weight), 1e-12)
            ),
        })

    return {
        "layers": layers,
        "global": {
            "edges": global_edges.tolist(),
            "orig_hist": global_orig.tolist(),
            "deq_hist": global_deq.tolist(),
            "count": total_weights,
            "xmin": GLOBAL_XMIN,
            "xmax": GLOBAL_XMAX,
        },
    }


def _log_y(counts_max: float, height: float, top: float, bottom: float):
    ymin = 0.5
    log_max = math.log10(counts_max)
    log_min = math.log10(ymin)

    def y_of(value: float) -> float:
        value = max(value, ymin)
        return top + (log_max - math.log10(value)) / (log_max - log_min) * (height - top - bottom)

    return y_of


def _global_svg(global_data, width: int, height: int):
    pl, pr, pt, pb = 46, 12, 8, 22
    edges = global_data["edges"]
    xmin, xmax = global_data["xmin"], global_data["xmax"]

    def x_of(value: float) -> float:
        return pl + (value - xmin) / (xmax - xmin) * (width - pl - pr)

    counts_max = max(max(global_data["orig_hist"]), max(global_data["deq_hist"]))
    y_of = _log_y(counts_max, height, pt, pb)
    base_y = pt + (height - pt - pb)

    def area_path(counts):
        parts = []
        for index, count in enumerate(counts):
            parts.append(f"{x_of(edges[index]):.2f},{y_of(count):.2f}")
            parts.append(f"{x_of(edges[index + 1]):.2f},{y_of(count):.2f}")
        return f"M {pl:.2f},{base_y:.2f} L " + " ".join(parts) + f" L {(width - pr):.2f},{base_y:.2f} Z"

    def line_path(counts):
        parts = []
        for index, count in enumerate(counts):
            parts.append(f"{x_of(edges[index]):.2f},{y_of(count):.2f}")
            parts.append(f"{x_of(edges[index + 1]):.2f},{y_of(count):.2f}")
        return "M " + " L ".join(parts)

    grid_lines = []
    tick = 1
    while tick <= counts_max * 10:
        if 1 <= tick <= counts_max:
            label = {
                1: "1", 10: "10", 100: "100", 1000: "1k", 10000: "10k",
                100000: "100k", 1000000: "1M", 10000000: "10M",
            }.get(tick, str(tick))
            grid_lines.append((y_of(tick), label))
        tick *= 10
    grid_svg = "".join(
        f'<line x1="{pl}" x2="{width - pr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>'
        f'<text x="{pl - 6}" y="{y + 3:.1f}" text-anchor="end" class="tick">{label}</text>'
        for y, label in grid_lines
    )
    x_tick_svg = "".join(
        f'<line x1="{x_of(t):.1f}" x2="{x_of(t):.1f}" y1="{base_y:.1f}" y2="{base_y + 6:.1f}" stroke="var(--line)"/>'
        f'<text x="{x_of(t):.1f}" y="{base_y + 18:.1f}" text-anchor="middle" class="tick">{t}</text>'
        for t in (-2, -1, 0, 1, 2, 3)
    )
    zero_x = x_of(0)

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Global weight histogram" class="plot">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="var(--surface)"/>'
        f'{grid_svg}'
        f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="8" y2="{base_y:.1f}" '
        f'stroke="var(--zero)" stroke-dasharray="2 3"/>'
        f'<path d="{area_path(global_data["orig_hist"])}" fill="var(--global-fill)" '
        f'stroke="var(--global-stroke)" stroke-width="1"/>'
        f'<path d="{line_path(global_data["deq_hist"])}" fill="none" '
        f'stroke="var(--after-stroke)" stroke-width="1.5" stroke-linejoin="round"/>'
        f'<line x1="{pl}" x2="{width - pr}" y1="{base_y:.1f}" y2="{base_y:.1f}" stroke="var(--line)"/>'
        f'{x_tick_svg}'
        f'<text x="{(pl + width - pr) / 2:.1f}" y="{height - 4}" text-anchor="middle" class="axis-title">'
        f'weight value (shared axis, log-count)</text>'
        f'</svg>'
    )


def _ridge_svg(layer, width: int, height: int) -> str:
    edges = layer["edges"]
    pl, pr = 10, 10

    def x_of(value: float) -> float:
        return pl + (value - edges[0]) / (edges[-1] - edges[0]) * (width - pl - pr)

    orig = layer["orig_hist"]
    deq = layer["deq_hist"]
    total_orig = max(sum(orig), 1)
    total_deq = max(sum(deq), 1)
    dens_orig = [c / total_orig for c in orig]
    dens_deq = [c / total_deq for c in deq]
    peak = max(max(dens_orig), max(dens_deq)) or 1e-9
    top_pad = 3
    bottom = height - 3

    def y_of(density: float) -> float:
        return top_pad + (1 - density / peak) * (height - top_pad - 3)

    pts_o, pts_d = [], []
    for index in range(len(orig)):
        x0 = x_of(edges[index])
        x1 = x_of(edges[index + 1])
        pts_o.append(f"{x0:.2f},{y_of(dens_orig[index]):.2f}")
        pts_o.append(f"{x1:.2f},{y_of(dens_orig[index]):.2f}")
        pts_d.append(f"{x0:.2f},{y_of(dens_deq[index]):.2f}")
        pts_d.append(f"{x1:.2f},{y_of(dens_deq[index]):.2f}")
    path_orig = f"M {pl:.2f},{bottom:.2f} L " + " ".join(pts_o) + f" L {(width - pr):.2f},{bottom:.2f} Z"
    path_deq = "M " + " L ".join(pts_d)

    return (
        f'<svg viewBox="0 0 {width} {height}" class="ridge">'
        f'<line x1="{x_of(0):.1f}" x2="{x_of(0):.1f}" y1="0" y2="{height}" '
        f'stroke="var(--zero)" stroke-dasharray="2 3" opacity="0.5"/>'
        f'<path d="{path_orig}" fill="var(--ridge-{layer["stage"]})" '
        f'stroke="var(--ridge-stroke-{layer["stage"]})" stroke-width="1"/>'
        f'<path d="{path_deq}" fill="none" stroke="var(--after-stroke)" '
        f'stroke-width="1.4" stroke-linejoin="round"/>'
        f'<text x="{pl + 2}" y="{height - 4}" class="axtick" text-anchor="start">'
        f'-{layer["max_abs"]:.2f}</text>'
        f'<text x="{width - pr - 2}" y="{height - 4}" class="axtick" text-anchor="end">'
        f'+{layer["max_abs"]:.2f}</text>'
        f'</svg>'
    )


CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --ink: #0b0b0b; --muted: #52514e;
  --line: #d5d3c8; --grid: #ecebe4; --zero: #a19f92;
  --global-fill: #cde2fb; --global-stroke: #2a78d6;
  --after-stroke: #eb6834; --after-swatch: #eb6834; --accent: #2a78d6;
  --ridge-stem:  #cde2fb;  --ridge-stroke-stem:  #86b6ef;
  --ridge-layer1: #b7d3f6; --ridge-stroke-layer1: #5598e7;
  --ridge-layer2: #9ec5f4; --ridge-stroke-layer2: #2a78d6;
  --ridge-layer3: #86b6ef; --ridge-stroke-layer3: #1c5cab;
  --ridge-layer4: #6da7ec; --ridge-stroke-layer4: #0d366b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface: #1a1a19; --ink: #ffffff; --muted: #c3c2b7;
    --line: #3c3a34; --grid: #2a2926; --zero: #7a776e;
    --global-fill: #184f95; --global-stroke: #3987e5;
    --after-stroke: #eb6834; --after-swatch: #eb6834; --accent: #3987e5;
    --ridge-stem:  #0d366b;  --ridge-stroke-stem:  #86b6ef;
    --ridge-layer1: #104281; --ridge-stroke-layer1: #5598e7;
    --ridge-layer2: #184f95; --ridge-stroke-layer2: #3987e5;
    --ridge-layer3: #1c5cab; --ridge-stroke-layer3: #86b6ef;
    --ridge-layer4: #256abf; --ridge-stroke-layer4: #cde2fb;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --ink: #ffffff; --muted: #c3c2b7;
  --line: #3c3a34; --grid: #2a2926; --zero: #7a776e;
  --global-fill: #184f95; --global-stroke: #3987e5;
  --after-stroke: #eb6834; --after-swatch: #eb6834; --accent: #3987e5;
  --ridge-stem:  #0d366b;  --ridge-stroke-stem:  #86b6ef;
  --ridge-layer1: #104281; --ridge-stroke-layer1: #5598e7;
  --ridge-layer2: #184f95; --ridge-stroke-layer2: #3987e5;
  --ridge-layer3: #1c5cab; --ridge-stroke-layer3: #86b6ef;
  --ridge-layer4: #256abf; --ridge-stroke-layer4: #cde2fb;
}
html, body { margin: 0; padding: 0; background: var(--surface); color: var(--ink); }
body { font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.fig { max-width: 820px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
.subtitle { margin: 0 0 16px; color: var(--muted); font-size: 13px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px 16px; margin: 0 0 20px; padding: 12px 14px;
  border: 1px solid var(--line); border-radius: 8px; }
.stats > div { display: flex; flex-direction: column; }
.stats dt { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.stats dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 500; }
.panel-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); margin: 20px 0 6px; }
.panel-note { color: var(--muted); font-size: 12px; margin: 4px 0 8px; }
.plot, .ridge { display: block; width: 100%; height: auto; overflow: visible; }
.tick, .axis-title, .axtick { fill: var(--muted); font-family: inherit; font-variant-numeric: tabular-nums; }
.tick { font-size: 10px; }
.axtick { font-size: 9px; }
.axis-title { font-size: 11px; }
.colhead { display: grid; grid-template-columns: 96px 1fr 44px 44px; gap: 6px;
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--muted); margin-bottom: 4px; }
.colhead .h { text-align: right; }
.colhead .h.plot { text-align: center; }
.rows { display: grid; gap: 2px; }
.row { display: grid; grid-template-columns: 96px 1fr 44px 44px; gap: 6px;
  align-items: center; font-variant-numeric: tabular-nums; }
.row .lbl { font-size: 11px; color: var(--muted); text-align: right;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .val { font-size: 11px; text-align: right; }
.row .cnt { font-size: 10px; color: var(--muted); text-align: right; }
.grow { display: grid; grid-template-columns: 96px 1fr 44px 44px; gap: 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px 18px; font-size: 12px;
  color: var(--muted); margin: 10px 0 0; align-items: center; }
.legend .sw { display: inline-block; width: 14px; height: 10px; border-radius: 2px;
  vertical-align: middle; margin-right: 6px; border: 1px solid transparent; }
.legend .lineswatch { display: inline-block; width: 18px; height: 0;
  border-top: 2px solid var(--after-swatch); vertical-align: middle; margin-right: 6px; }
@media (max-width: 480px) {
  .row, .grow, .colhead { grid-template-columns: 76px 1fr 40px 40px; }
  .row .lbl { font-size: 10px; }
}
"""


def _render(data) -> str:
    global_svg = _global_svg(data["global"], 620, 240)

    rows = []
    for layer in data["layers"]:
        svg = _ridge_svg(layer, 620, 40)
        count = layer["count"]
        count_str = f"{count / 1e3:.0f}k" if count < 1e6 else f"{count / 1e6:.2f}M"
        step = layer["channel_scale_median"]
        step_str = f"{step * 1000:.2f}m" if step < 0.01 else f"{step:.3f}"
        err_str = f"{layer['rms_error_relative'] * 100:.2f}%"
        rows.append(
            f'<div class="row" data-stage="{layer["stage"]}" '
            f'title="{layer["name"]} · max|W| {layer["max_abs"]:.4f} · '
            f'step {step:.6f} · rms err {err_str}">'
            f'<span class="lbl">{layer["name"]}</span>{svg}'
            f'<span class="val">{step_str}</span>'
            f'<span class="cnt">{err_str}</span>'
            f'</div>'
        )
    rows_html = "".join(rows)

    total = data["global"]["count"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ResNet-18 weight distribution</title>
<style>{CSS}</style>
</head>
<body>
<figure class="fig">
  <h1>ResNet-18 weight distribution — before and after INT8 quantization</h1>
  <p class="subtitle">Blue fill: original density after BatchNorm folding.
  Orange line: density after INT8 per-output-channel quantization then
  dequantization. Each layer's x-axis is scaled to its own ±max|W|.</p>
  <dl class="stats">
    <div><dt>total weights</dt><dd>{total:,}</dd></div>
    <div><dt>quantization</dt><dd>INT8, per output channel</dd></div>
    <div><dt>global RMS error</dt><dd>&lt; 2%</dd></div>
  </dl>
  <p class="panel-title">A — Global histogram (shared axis, log count)</p>
  <div class="grow"><span></span>{global_svg}<span></span><span></span></div>
  <p class="panel-title">B — Per-layer density (independent x per layer)</p>
  <p class="panel-note">Right columns: median channel quantization step
  (<code>0.5m</code> = 0.0005) and relative RMS reconstruction error.</p>
  <div class="colhead">
    <span class="h">layer</span>
    <span class="h plot">-max|W| &larr; 0 &rarr; +max|W|</span>
    <span class="h">step</span><span class="h">rms err</span>
  </div>
  <div class="rows">{rows_html}</div>
  <div class="legend">
    <span><span class="sw" style="background:var(--global-fill);border-color:var(--global-stroke)"></span>original weights</span>
    <span><span class="lineswatch"></span>INT8 quantized then dequantized</span>
  </div>
</figure>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    data = _compute(arguments.checkpoint)
    html = _render(data)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(html, encoding="utf-8")
    print(f"PASS: wrote {arguments.output} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
