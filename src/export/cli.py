"""Export a model into a package: export() runs the front end (pipeline.py)
and the compiler (compile.py). The command line runs it, or one front-end
stage at a time.

    python -m src.export.cli build src/export/configs/smollm.yml --out dist/smollm/package
    python -m src.export.cli export smollm --out build/pkg/smollm [--bits 4 --kv int8 ...]
    python -m src.export.cli load smollm --source <dir> --out build/stage/source.json
    python -m src.export.cli quantize build/stage/source.json --bits 4 --out build/stage/quantized
    python -m src.export.cli lower build/stage/quantized --chunk 16 --out build/stage/graph
    python -m src.export.cli prepare-onnx prepare --model-dir <onnx model>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.export import model_arch, pipeline
from src.export.compile import compile_graph
from src.export.ir.graph import ModelGraph
from src.export.ir.options import LLMOptions, QuantRecipe, ShapeOptions
from src.export.pipeline import build_graph
from src.runtime.core.package import model_env
from src.vendors.npu_pynq.isa import isa


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


def _options(args) -> LLMOptions:
    return LLMOptions(chunk=args.chunk, max_seq=args.max_seq, bits=args.bits, group=args.group,
                      embed_bits=args.embed_bits, kv=args.kv, attention=args.attention, awq=args.awq,
                      layers=args.layers)


def cmd_export(args) -> int:
    if args.model == "resnet18":
        from src.export.model_arch.resnet18.arch import ResNetOptions
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
    from src.export.model_arch.smollm.onnx import ONNX_PIN
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


def cmd_build(args) -> int:
    config = pipeline.read_config(args.config)
    timings: dict[str, float] = {}
    out = pipeline.export_package(config, args.out, source=args.source, timings=timings)
    print(json.dumps({'package': str(out), 'seconds': {k: round(v, 1) for k, v in timings.items()}}, indent=2))
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

    b = sub.add_parser("build", help="Export through PyTorch from a configs/*.yml file")
    b.add_argument("config")
    b.add_argument("--out", required=True)
    b.add_argument("--source", help="Override the config's model source")
    b.set_defaults(fn=cmd_build)
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
