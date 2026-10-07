"""Compile a model graph into a package for the host and the PYNQ-Z1.

    model frontend -> MLIR (linalg on tensors, npu.weight)
      npu-opt --npu-partition        which contractions become NPU tasks
      pack weights                   B tiles for NPU-read weights, dense otherwise
      npu-opt --npu-lower            ISA programs + runtime calls
      mlir-opt / mlir-translate / llc / link, once per target

The package directory is self-contained: ``CompiledModel`` (src/runtime)
loads it on either platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import numpy as np

from src.compiler import toolchain
from src.compiler.mlir_builder import ModuleBuilder
from src.isa import isa, layout

FORMAT = "npu-package/1"
WEIGHT_ALIGN = 64


@dataclass
class ModelGraph:
    """What a model frontend hands to the compiler."""

    name: str
    module: ModuleBuilder
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    entry_points: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)  # copied verbatim (tokenizer, ...)

    def add_weight(self, name: str, array: np.ndarray) -> str:
        if name in self.weights:
            raise ValueError(f"duplicate weight {name}")
        self.weights[name] = np.ascontiguousarray(array)
        return name


def pack_weights(graph: ModelGraph, report: dict, limits: isa.Limits) -> tuple[bytes, dict, dict]:
    """Return (blob, {name: offset}, {name: layout}) with NPU weights tiled."""
    npu_layout: dict[str, tuple[int, int, int]] = {}
    for task in report["npu_tasks"]:
        key = (task["k"], task["n"], task["tk"])
        prev = npu_layout.setdefault(task["weight"], key)
        if prev != key:
            raise ValueError(f"weight {task['weight']} used by NPU tasks with different tiling")
    for name, (k, n, tk) in graph.meta.get("require_npu_layout", {}).items():
        if npu_layout.get(name) != (k, n, tk):
            raise ValueError(f"{name} is read in NPU layout (k={k}, n={n}, tk={tk}) but the partition "
                             f"stores it as {npu_layout.get(name, 'dense')}")
    packed = set(graph.meta.get("packed_int4", []))
    chunks: list[bytes] = []
    offsets: dict[str, int] = {}
    layouts: dict[str, str] = {}
    cursor = 0
    for name, array in graph.weights.items():
        if name in npu_layout:
            k, n, tk = npu_layout[name]
            if array.shape != (k, n) or array.dtype != np.int8:
                raise ValueError(f"NPU weight {name} must be int8 {k}x{n}, got {array.dtype} {array.shape}")
            plan = layout.plan_gemm(1, n, k, limits, tk)
            tiles = layout.pack_b(plan, array)
            if name in packed:
                if tiles.min() < -8 or tiles.max() > 7:
                    raise ValueError(f"{name} is marked INT4 but holds values outside [-8, 7]")
                u = tiles.view(np.uint8) & 0x0F
                data = (u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()
                layouts[name] = f"npu_b4(k={k},n={n},tk={tk})"
            else:
                data = tiles.tobytes()
                layouts[name] = f"npu_b(k={k},n={n},tk={tk})"
        else:
            data = np.ascontiguousarray(array).tobytes()
            layouts[name] = f"dense({array.dtype},{'x'.join(map(str, array.shape))})"
        pad = (-cursor) % WEIGHT_ALIGN
        chunks.append(b"\0" * pad)
        cursor += pad
        offsets[name] = cursor
        chunks.append(data)
        cursor += len(data)
    return b"".join(chunks), offsets, layouts


def export(graph: ModelGraph, out: str | Path, targets: Sequence[str] = ("host", "pynq"),
           limits: isa.Limits = isa.Limits(), keep_ir: bool = True, segment_bytes: int = 4 << 20) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    work = out / "ir"
    work.mkdir(parents=True)
    timings: dict[str, float] = {}

    t = time.monotonic()
    source = work / "model.mlir"
    source.write_text(graph.module.render())
    parted = work / "partitioned.mlir"
    report_path = out / "partition.json"
    toolchain.partition(source, parted, report_path, max_k=limits.max_k)
    report = json.loads(report_path.read_text())
    timings["partition"] = time.monotonic() - t

    t = time.monotonic()
    blob, offsets, layouts = pack_weights(graph, report, limits)
    (out / "weights.bin").write_bytes(blob)
    table = work / "weights.json"
    table.write_text(json.dumps(offsets))
    timings["pack_weights"] = time.monotonic() - t

    t = time.monotonic()
    lowered = work / "lowered.mlir"
    toolchain.lower(parted, lowered, table, out / "programs", limits.rows, limits.columns, limits.max_k,
                    segment_bytes)
    timings["npu_lower"] = time.monotonic() - t

    libraries = {}
    for name in targets:
        t = time.monotonic()
        tgt = toolchain.target(name)
        lib = toolchain.codegen(lowered, work / name, tgt)
        dest = out / name / tgt.library
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(lib, dest)
        libraries[name] = f"{name}/{tgt.library}"
        timings[f"codegen_{name}"] = time.monotonic() - t

    for dest, src in graph.files.items():
        shutil.copy2(src, out / dest)
    programs = json.loads((out / "programs.json").read_text())
    manifest = {
        "format": FORMAT,
        "model": graph.name,
        "libraries": libraries,
        "entry_points": graph.entry_points,
        "meta": graph.meta,
        "weights": {"file": "weights.bin", "bytes": len(blob), "count": len(offsets)},
        "weight_layouts": layouts,
        "weight_offsets": offsets,
        "npu": {"tasks": len(programs["tasks"]),
                "segments": sum(len(t["segments"]) for t in programs["tasks"]),
                "program_words": programs["words"],
                "jobs_total": sum(t["jobs"] for t in programs["tasks"])},
        "cpu_linalg_ops": report["cpu_linalg_ops"],
        "limits": {"rows": limits.rows, "columns": limits.columns, "max_k": limits.max_k},
        "compile_seconds": timings,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not keep_ir:
        shutil.rmtree(work)
    else:
        for name in targets:
            for f in (work / name).glob("*.o"):
                f.unlink()
    return out
