"""Compile a model into a package for the host and the PYNQ-Z1, and the
command line that drives it.

    model frontend (src/export/graphs) -> MLIR (linalg on tensors, npu.weight)
      npu-opt --npu-partition        which contractions become NPU tasks
      pack weights                   B tiles for NPU-read weights, dense otherwise
      npu-opt --npu-lower            ISA programs + runtime calls
      mlir-opt / mlir-translate / llc / link, once per target

The package directory is self-contained: ``Package`` (src/runtime) loads it
on either platform.

    python -m src.export.cli export smollm --out build/pkg/smollm [--bits 4 --kv int8 ...]
    python -m src.export.cli prepare-onnx prepare --model-dir <onnx model>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Sequence

from src.export.core.graph import LLMOptions, ModelGraph
from src.export.targets import npu_opt, target
from src.vendors.npu_pynq.compiler.package import pack_weights
from src.vendors.npu_pynq.compiler.passes import RUNTIME_C
from src.export.compilers.mlir import toolchain
from src.vendors.npu_pynq.isa import isa
from src.runtime.core.package import model_env, onnx_asset_dir

FORMAT = "npu-package/1"


def export(graph: ModelGraph | str | Path | None, out: str | Path, targets: Sequence[str] | None = None,
           limits: isa.Limits = isa.Limits(), keep_ir: bool = True, segment_bytes: int = 4 << 20,
           extra_objects: dict[str, Sequence[Path]] | None = None, *, model: str | None = None,
           opts=None, source_format: str = 'auto', prefill_lengths=(36, 40, 42),
           native_object: str | Path | None = None) -> Path:
    """Compile a graph or model source through the same partition/lower pipeline.

    Model frontends own input formats; the compiler always consumes ModelGraph.
    Existing graph callers retain their original API and default CPU targets.
    """
    frontend_seconds = None
    if not isinstance(graph, ModelGraph):
        from src.export.graphs import load
        if model is None:
            raise ValueError('A model frontend is required when exporting a source')
        kwargs = {} if opts is None else {'opts': opts}
        if graph is not None:
            kwargs['source'] = graph
        if source_format != 'auto':
            kwargs['source_format'] = source_format
        source_path = Path(graph) if graph is not None else None
        if source_format == 'onnx' or (source_path is not None and
                (source_path.suffix == '.onnx' or (source_path/'model_int8.onnx').is_file()
                 or (onnx_asset_dir(source_path, 'onnx')/'model_int8.onnx').is_file())):
            kwargs['prefill_lengths'] = prefill_lengths
        start = time.monotonic()
        graph, _ = load(model).build(**kwargs)
        frontend_seconds = time.monotonic() - start
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
    timings: dict[str, float] = {}
    if frontend_seconds is not None:
        timings['frontend'] = frontend_seconds

    # Write the model graph as MLIR, then choose which matmuls run on the NPU.
    t = time.monotonic()
    source = work / "model.mlir"
    source.write_text(graph.module.render())

    parted = work / "partitioned.mlir"
    report_path = out / "partition.json"
    npu_opt.partition(source, parted, report_path, max_k=limits.max_k)
    report = json.loads(report_path.read_text())
    timings["partition"] = time.monotonic() - t

    # Pack NPU weights as tiles and record every weight's offset in the arena.
    t = time.monotonic()
    blob, offsets, layouts = pack_weights(graph.weights, graph.meta, report, limits)
    (out / "weights.bin").write_bytes(blob)
    table = work / "weights.json"
    table.write_text(json.dumps(offsets))
    timings["pack_weights"] = time.monotonic() - t

    # Turn NPU tasks into ISA programs and calls to the C runtime.
    t = time.monotonic()
    lowered = work / "lowered.mlir"
    npu_opt.lower(parted, lowered, table, out / "programs", limits.rows, limits.columns, limits.max_k,
                 segment_bytes)
    timings["npu_lower"] = time.monotonic() - t

    # Compile the remaining CPU operations for each requested platform.
    libraries = {}
    for name in targets:
        t = time.monotonic()
        tgt = target(name)
        lib = toolchain.codegen(lowered, work / name, tgt, objects.get(name, ()), runtime_c=RUNTIME_C)
        dest = out / name / tgt.library
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(lib, dest)
        libraries[name] = f"{name}/{tgt.library}"
        timings[f"codegen_{name}"] = time.monotonic() - t

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
    from src.export.graphs import load
    if args.model == "resnet18":
        opts = load(args.model).ResNetOptions(bits=args.bits)
    else:
        opts = _options(args)
    pkg = export(args.source or args.model_dir, args.out, model=args.model, opts=opts,
                 source_format=args.format, native_object=args.native_object,
                 prefill_lengths=tuple(map(int, args.prefill_lengths.split(','))),
                 targets=args.targets.split(',') if args.targets else None)
    manifest = json.loads((pkg / "manifest.json").read_text())
    print(json.dumps({"package": str(pkg), "weights_bytes": manifest["weights"]["bytes"],
                      "npu_tasks": manifest["npu"]["tasks"], "compile_seconds": manifest["compile_seconds"]}))
    return 0


def cmd_prepare_onnx(args) -> int:
    from src.export.core.onnx_source import fetch, export_graph, verify_export, export_operators
    from src.export.graphs.smollm.onnx import ONNX_PIN
    root = Path(args.model_dir).expanduser().resolve()
    if args.phase in ("fetch", "prepare"):
        fetch(root, ONNX_PIN)
    if args.phase in ("export", "prepare"):
        export_graph(root, ONNX_PIN)
    if args.phase in ("verify", "prepare"):
        print(json.dumps(verify_export(root), indent=2))
    if args.phase in ("operators", "prepare"):
        export_operators(root)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="src.export.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("model")
    e.add_argument("--format", type=lambda s: {'checkpoint':'auto', 'official-onnx':'onnx'}.get(s, s),
                   choices=('auto', 'safetensors', 'onnx'), default='auto')
    e.add_argument("--model-dir", default=model_env("NPU_ONNX_MODEL"))
    e.add_argument("--native-object")
    e.add_argument("--prefill-lengths", default="36,40,42")
    e.add_argument("--layers", type=int)
    e.add_argument("--source")
    e.add_argument("--out", required=True)
    e.add_argument("--targets")
    e.add_argument("--chunk", type=int, default=16)
    e.add_argument("--max-seq", type=int, default=512)
    e.add_argument("--bits", type=int, default=8)
    e.add_argument("--group", type=int, default=128)
    e.add_argument("--embed-bits", type=int, default=8)
    e.add_argument("--kv", default="f32", choices=("f32", "int8"))
    e.add_argument("--attention", default="online", choices=("online", "naive"))
    e.add_argument("--awq", action="store_true")
    e.set_defaults(fn=cmd_export)
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
