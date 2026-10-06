"""Command line for exporting and running models.

    python -m src.inference.cli export smollm --out build/pkg/smollm [--bits 4 --kv int8 ...]
    python -m src.inference.cli run build/pkg/smollm --prompt "Hello" [--chat --max-new 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from src.inference.graph import LLMOptions


def _options(args) -> LLMOptions:
    return LLMOptions(chunk=args.chunk, max_seq=args.max_seq, bits=args.bits, group=args.group,
                      embed_bits=args.embed_bits, kv=args.kv, attention=args.attention, awq=args.awq)


def cmd_export(args) -> int:
    from src.compiler.export import export
    from src.models import load

    module = load(args.model)
    t = time.monotonic()
    kwargs = {}
    if args.source:
        kwargs["source"] = args.source
    if hasattr(module, "LLM") or args.model != "resnet18":
        kwargs["opts"] = _options(args)
    graph, _ = module.build(**kwargs)
    built = time.monotonic() - t
    pkg = export(graph, args.out, targets=args.targets.split(","))
    manifest = json.loads((pkg / "manifest.json").read_text())
    manifest["compile_seconds"]["frontend"] = built
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"package": str(pkg), "weights_bytes": manifest["weights"]["bytes"],
                      "npu_tasks": manifest["npu"]["tasks"], "compile_seconds": manifest["compile_seconds"]}))
    return 0


def cmd_run(args) -> int:
    from src.inference.engine import LLMEngine
    from src.inference.sampling import SamplingParams

    overlay = None
    if args.backend == "pynq":
        from pynq import Overlay
        overlay = Overlay(args.overlay)
    engine = LLMEngine(args.package, args.backend, overlay=overlay)
    params = SamplingParams(args.temperature, args.top_k, args.top_p, args.min_p, args.repetition_penalty, args.seed)

    def show(_tid, piece):
        sys.stdout.write(piece)
        sys.stdout.flush()

    ids, text, stats = engine.generate(args.prompt, args.max_new, params, chat=args.chat,
                                       on_token=show if args.stream else None)
    if not args.stream:
        print(text)
    print()
    report = {"prompt": args.prompt, "output": text, "output_ids": ids, **stats.as_dict()}
    print(json.dumps(stats.as_dict()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="src.inference.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("model")
    e.add_argument("--source")
    e.add_argument("--out", required=True)
    e.add_argument("--targets", default="host,pynq")
    e.add_argument("--chunk", type=int, default=16)
    e.add_argument("--max-seq", type=int, default=512)
    e.add_argument("--bits", type=int, default=8)
    e.add_argument("--group", type=int, default=128)
    e.add_argument("--embed-bits", type=int, default=8)
    e.add_argument("--kv", default="f32", choices=("f32", "int8"))
    e.add_argument("--attention", default="online", choices=("online", "naive"))
    e.add_argument("--awq", action="store_true")
    e.set_defaults(fn=cmd_export)
    r = sub.add_parser("run")
    r.add_argument("package")
    r.add_argument("--prompt", required=True)
    r.add_argument("--chat", action="store_true")
    r.add_argument("--max-new", type=int, default=32)
    r.add_argument("--backend", default=None)
    r.add_argument("--overlay", default="overlay/artifacts/npu_matrix.bit")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--top-k", type=int, default=0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--min-p", type=float, default=0.0)
    r.add_argument("--repetition-penalty", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--stream", action="store_true")
    r.add_argument("--json")
    r.set_defaults(fn=cmd_run)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
