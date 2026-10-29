"""Compile a model into a package for the host and the PYNQ-Z1, and the
command line that drives it.

    front end (src/export/pipeline.py)  load -> quantize -> to_mlir: MLIR (linalg on tensors, npu.weight)
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

    python -m src.export.cli export smollm --out build/pkg/smollm [--bits 4 --kv int8 ...]

or one front-end stage at a time:

    python -m src.export.cli load smollm --source <dir> --out build/stage/source.json
    python -m src.export.cli quantize build/stage/source.json --bits 4 --out build/stage/quantized
    python -m src.export.cli lower build/stage/quantized --chunk 16 --out build/stage/graph
    python -m src.export.cli prepare-onnx prepare --model-dir <onnx model>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from src.export.ir.graph import ModelGraph
from src.export import model_arch, pipeline
from src.export.ir.options import LLMOptions, QuantRecipe, ShapeOptions
from src.export.pipeline import build_graph
from src.export.targets import npu_opt, target
from src.vendors.npu_pynq.compiler.package import pack_weights
from src.vendors.npu_pynq.compiler.passes import RUNTIME_C
from src.export.compilers.mlir import toolchain
from src.vendors.npu_pynq.isa import isa
from src.runtime.core.package import model_env

FORMAT = "npu-package/1"


def export(graph: ModelGraph | str | Path | None, out: str | Path, targets: Sequence[str] | None = None,
           limits: isa.Limits = isa.Limits(), keep_ir: bool = True, segment_bytes: int = 4 << 20,
           extra_objects: dict[str, Sequence[Path]] | None = None, *, model: str | None = None,
           opts=None, source_format: str = 'auto', prefill_lengths=(36, 40, 42),
           native_object: str | Path | None = None, onnx_import_dir: str | Path | None = None) -> Path:
    """Compile a graph or model source through the same partition/lower pipeline.

    A source runs the front end first (load, quantize, to_mlir in
    src/export/pipeline.py); the compiler always consumes ModelGraph.
    Existing graph callers retain their original API and default CPU targets.
    """
    timings: dict[str, float] = {}
    if not isinstance(graph, ModelGraph):
        if model is None:
            raise ValueError('A model frontend is required when exporting a source')
        graph, _ = build_graph(model, graph, opts, source_format=source_format, prefill_lengths=prefill_lengths,
                               onnx_import_dir=onnx_import_dir, timings=timings)
        timings['frontend'] = timings['load'] + timings['quantize'] + timings['to_mlir']
    return compile_graph(graph, out, targets, limits, keep_ir, segment_bytes, extra_objects,
                         native_object=native_object, timings=timings)


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


def _options(args) -> LLMOptions:
    return LLMOptions(chunk=args.chunk, max_seq=args.max_seq, bits=args.bits, group=args.group,
                      embed_bits=args.embed_bits, kv=args.kv, attention=args.attention, awq=args.awq,
                      layers=args.layers)


def cmd_export(args) -> int:
    if args.model == "resnet18":
        from src.export.model_arch.resnet18 import ResNetOptions
        opts = ResNetOptions(bits=args.bits)
    else:
        opts = _options(args)
    pkg = export(args.source or args.model_dir, args.out, model=args.model, opts=opts,
                 source_format=args.format, native_object=args.native_object, onnx_import_dir=args.onnx_import_dir,
                 prefill_lengths=tuple(map(int, args.prefill_lengths.split(','))),
                 targets=args.targets.split(',') if args.targets else None)
    manifest = json.loads((pkg / "manifest.json").read_text())
    print(json.dumps({"package": str(pkg), "weights_bytes": manifest["weights"]["bytes"],
                      "npu_tasks": manifest["npu"]["tasks"], "compile_seconds": manifest["compile_seconds"]}))
    return 0


def _prefill_lengths(args) -> tuple[int, ...]:
    return tuple(map(int, args.prefill_lengths.split(',')))


def cmd_load(args) -> int:
    """Read and verify a source; record how to read it again."""
    src = pipeline.load(args.model, args.source or args.model_dir, args.format, args.layers,
                        onnx_import_dir=args.onnx_import_dir)
    record = {"model": args.model, "source": src.source, "format": src.format, "layers": args.layers,
              "onnx_import_dir": args.onnx_import_dir, "arch": src.arch, "root": str(src.root)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    return 0


def cmd_quantize(args) -> int:
    """Replay a recorded load, quantize, and save the parameters."""
    record = json.loads(Path(args.source_record).read_text())
    arch, _ = pipeline.architecture(record["model"], record["source"], record["format"])
    recipe = QuantRecipe(args.bits, args.group, args.embed_bits, args.awq)
    arch.validate(recipe, ShapeOptions(), None)
    src = pipeline.load(record["model"], record["source"], record["format"], record["layers"],
                        onnx_import_dir=record["onnx_import_dir"])
    out = pipeline.save_quantized(pipeline.quantize(src, recipe), args.out)
    print(json.dumps({"quantized": str(out), "arch": src.arch}))
    return 0


def cmd_lower(args) -> int:
    """Build the ModelGraph from saved parameters and write it out."""
    q = pipeline.load_quantized(args.quantized)
    shape = ShapeOptions(args.chunk, args.max_seq, args.attention, args.kv, _prefill_lengths(args))
    model_arch.get(q.source.arch).validate(q.recipe, shape, None)
    out = pipeline.save_graph(pipeline.to_mlir(q, shape), args.out)
    print(json.dumps({"graph": str(out), "mlir": str(out / "model.mlir")}))
    return 0


def cmd_prepare_onnx(args) -> int:
    from src.export.load.onnx_prepare import fetch, export_graph, verify_export, export_operators
    from src.export.model_arch.smollm_onnx import ONNX_PIN
    root = Path(args.model_dir).expanduser().resolve()
    if args.phase in ("fetch", "prepare"):
        fetch(root, ONNX_PIN)
    if args.phase in ("export", "prepare"):
        export_graph(root, ONNX_PIN)
    if args.phase in ("verify", "prepare"):
        print(json.dumps(verify_export(root), indent=2))
    if args.phase == "operators":
        export_operators(root)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="src.export.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def source_args(p):
        p.add_argument("model")
        p.add_argument('--onnx-import-dir', help='Verified ONNX-MLIR import directory; otherwise import automatically')
        p.add_argument("--format", type=lambda s: {'checkpoint':'auto', 'official-onnx':'onnx'}.get(s, s),
                       choices=('auto', 'safetensors', 'onnx', 'torch'), default='auto')
        p.add_argument("--model-dir", default=model_env("NPU_ONNX_MODEL"))
        p.add_argument("--layers", type=int)
        p.add_argument("--source")

    def recipe_args(p):
        p.add_argument("--bits", type=int, default=8)
        p.add_argument("--group", type=int, default=128)
        p.add_argument("--embed-bits", type=int, default=8)
        p.add_argument("--awq", action="store_true")

    def shape_args(p):
        p.add_argument("--prefill-lengths", default="36,40,42")
        p.add_argument("--chunk", type=int, default=16)
        p.add_argument("--max-seq", type=int, default=512)
        p.add_argument("--kv", default="f32", choices=("f32", "int8"))
        p.add_argument("--attention", default="online", choices=("online", "naive"))

    e = sub.add_parser("export", help="Compile a model source into a package")
    source_args(e)
    recipe_args(e)
    shape_args(e)
    e.add_argument("--native-object")
    e.add_argument("--out", required=True)
    e.add_argument("--targets")
    e.set_defaults(fn=cmd_export)
    s = sub.add_parser("load", help="Stage 1: read and verify a source; write a replayable record")
    source_args(s)
    s.add_argument("--out", required=True, help="record file (JSON)")
    s.set_defaults(fn=cmd_load)
    q = sub.add_parser("quantize", help="Stage 2: quantize a recorded source; write the parameters")
    q.add_argument("source_record")
    recipe_args(q)
    q.add_argument("--out", required=True, help="directory for params.npz and model.json")
    q.set_defaults(fn=cmd_quantize)
    lo = sub.add_parser("lower", help="Stage 3: build the ModelGraph (to_mlir) from saved parameters")
    lo.add_argument("quantized")
    shape_args(lo)
    lo.add_argument("--out", required=True, help="directory for model.mlir, weights.npz and graph.json")
    lo.set_defaults(fn=cmd_lower)
    o = sub.add_parser("prepare-onnx", help="Fetch and verify the pinned ONNX model")
    o.add_argument("phase", choices=("fetch", "export", "verify", "operators", "prepare"))
    o.add_argument("--model-dir", default=model_env("NPU_ONNX_MODEL"))
    o.set_defaults(fn=cmd_prepare_onnx)
    args = parser.parse_args(argv)
    if args.cmd == "prepare-onnx" and not args.model_dir:
        parser.error("Set --model-dir or NPU_ONNX_MODEL")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
