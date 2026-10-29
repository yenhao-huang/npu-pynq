"""Compile a ModelGraph into a package for the host and the PYNQ-Z1.

    ModelGraph (from pipeline.py: load -> quantize -> to_mlir): MLIR (linalg on tensors, npu.weight)
      npu-opt --npu-partition        which contractions become NPU tasks; split into
                                     an NPU subgraph (npu.module) and a host subgraph
      pack weights                   B tiles for NPU-read weights, dense otherwise
      NPU subgraph:  npu-opt --npu-encode       tasks -> ISA programs
      host subgraph: npu-opt --npu-lower-host   npu.call -> npu_rt_gemm(task id, ...)
                     mlir-opt / mlir-translate / llc / link, once per target

The two subgraph pipelines share only the task signatures and ids, and run
concurrently. They meet at run time, when npu_rt_gemm(id) runs program id.

The package directory is self-contained: ``Package`` (src/runtime) loads it
on either platform.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
from typing import Sequence

from src.export.compilers.mlir import toolchain
from src.export.ir.graph import ModelGraph
from src.export.targets import npu_opt, target
from src.vendors.npu_pynq.compiler.package import pack_weights
from src.vendors.npu_pynq.compiler.passes import RUNTIME_C
from src.vendors.npu_pynq.isa import isa

FORMAT = "npu-package/1"


def compile_graph(graph: ModelGraph, out: str | Path, targets: Sequence[str] | None = None,
                  limits: isa.Limits = isa.Limits(), keep_ir: bool = True, segment_bytes: int = 4 << 20,
                  extra_objects: dict[str, Sequence[Path]] | None = None, *,
                  native_object: str | Path | None = None, timings: dict[str, float] | None = None) -> Path:
    """Partition, pack, encode and generate code for ``graph``; write the package to ``out``."""
    supported = graph.meta.get('targets', ('host', 'pynq'))
    targets = tuple(targets) if targets is not None else tuple(supported)
    if not targets or any(t not in supported for t in targets):
        raise ValueError(f'Unsupported targets {targets}; this graph supports {supported}')
    objects = dict(extra_objects or {})
    if native_object is not None:
        objects['pynq'] = (*objects.get('pynq', ()), Path(native_object))
    for name in graph.meta.get('requires_native_object', ()):
        if name in targets and not objects.get(name):
            raise ValueError(f'{name} requires a native kernel object')
    for paths in objects.values():
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    out = Path(out)
    assets = list(graph.files.values()) + [p for paths in objects.values() for p in paths]
    if any(out.resolve() == Path(p).resolve() or out.resolve() in Path(p).resolve().parents
           for p in assets):
        raise ValueError('The output directory must not contain input assets')
    if out.exists():
        shutil.rmtree(out)
    work = out / "ir"
    work.mkdir(parents=True)
    timings = dict(timings or {})

    # Write the model graph as MLIR, choose which matmuls run on the NPU, and
    # split the module into the NPU and host subgraphs.
    t = time.monotonic()
    source = work / "model.mlir"
    source.write_text(graph.module.render())

    parted = work / "partitioned.mlir"
    report_path = out / "partition.json"
    npu_opt.partition(source, parted, report_path, max_k=limits.max_k)
    report = json.loads(report_path.read_text())
    timings["partition"] = time.monotonic() - t

    # Pack NPU weights as tiles and record every weight's offset in the arena.
    # Both subgraphs read the offsets: the NPU programs for B tiles, the host
    # code for every other weight.
    t = time.monotonic()
    blob, offsets, layouts = pack_weights(graph.weights, graph.meta, report, limits)
    (out / "weights.bin").write_bytes(blob)
    table = work / "weights.json"
    table.write_text(json.dumps(offsets))
    timings["pack_weights"] = time.monotonic() - t

    def npu_branch() -> dict[str, float]:
        # NPU subgraph: each task function becomes an ISA program.
        t = time.monotonic()
        npu_opt.encode(parted, work / "npu.mlir", table, out / "programs", limits.rows, limits.columns,
                       limits.max_k, segment_bytes)
        return {"npu_encode": time.monotonic() - t}

    def host_branch() -> tuple[dict[str, float], dict[str, str]]:
        # Host subgraph: task calls become runtime calls, then CPU code per platform.
        times = {}
        t = time.monotonic()
        host = work / "host.mlir"
        npu_opt.lower_host(parted, host, table)
        times["host_lower"] = time.monotonic() - t
        libraries = {}
        for name in targets:
            t = time.monotonic()
            tgt = target(name)
            lib = toolchain.codegen(host, work / name, tgt, objects.get(name, ()), runtime_c=RUNTIME_C)
            dest = out / name / tgt.library
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(lib, dest)
            libraries[name] = f"{name}/{tgt.library}"
            times[f"codegen_{name}"] = time.monotonic() - t
        return times, libraries

    with ThreadPoolExecutor(max_workers=2) as pool:
        npu_job = pool.submit(npu_branch)
        host_job = pool.submit(host_branch)
        timings.update(npu_job.result())
        host_times, libraries = host_job.result()
        timings.update(host_times)

    # Bundle the compiled libraries, weights, programs, and model metadata.
    for dest, src in graph.files.items():
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
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
