"""Command line for running packages and the ONNX reference on the host or board.

    python -m src.runtime.cli run build/pkg/smollm --prompt "Hello" [--chat --max-new 32]
    python -m src.runtime.cli reference --model-dir <onnx model> --prompt "Hello"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from src.runtime.core.package import model_env, onnx_asset_dir


def cmd_run(args) -> int:
    from src.runtime.core.engine import LLMEngine
    from src.runtime.core.sampling import SamplingParams
    from src.runtime.core.package import platform_backend

    backend = args.backend or platform_backend()
    overlay = None
    if backend == "pynq":
        from pynq import Overlay
        overlay = Overlay(args.overlay)
    engine = LLMEngine(args.package_override or args.package, backend, overlay=overlay)
    params = SamplingParams(args.temperature, args.top_k, args.top_p, args.min_p, args.repetition_penalty, args.seed)

    def show(_tid, piece):
        sys.stdout.write(piece)
        sys.stdout.flush()

    ids, text, stats = engine.generate(args.prompt, args.max_new, params, chat=args.chat,
                                       on_token=show if args.stream else None)
    stats = stats.as_dict()
    stats['runtime'] = engine.package.stats()
    if not args.stream:
        print(text)
    print()
    report = {"prompt": args.prompt, "output": text, "output_ids": ids, **stats}
    print(json.dumps(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


def cmd_reference(args) -> int:
    from src.test.model.onnx_reference import ONNXReferenceEngine
    from src.runtime.cpu.ort import OrtOperators, NativeOrtOperators
    root = Path(args.model_dir).expanduser().resolve()
    if not (onnx_asset_dir(root, "onnx-graph")).is_dir():
        root /= "packages"
    gemm = float_ops = None
    try:
        if args.backend == "pynq":
            from pynq import Overlay
            from src.vendors.npu_pynq.driver.isa import NpuGemm
            runtime = onnx_asset_dir(root, 'onnx-runtime')
            bridge = runtime / 'libonnx_ort_bridge.so'
            if not bridge.exists():
                bridge = runtime / 'libofficial_ort_bridge.so'
            float_ops = NativeOrtOperators(onnx_asset_dir(root, 'onnx-ops'), bridge)
            gemm = NpuGemm(Overlay(args.overlay))
        else:
            float_ops = OrtOperators(onnx_asset_dir(root, "onnx-ops"))
        engine = ONNXReferenceEngine(root, integer_gemm=gemm, float_ops=float_ops)
        ids, text, stats = engine.generate(args.prompt, args.max_new_tokens,
            chat=not args.raw, on_token=lambda _, piece: print(piece, end="", flush=True))
        print()
        print(json.dumps({"output_ids": ids, **stats,
                          "npu": gemm.stats if gemm is not None else {}}))
    finally:
        if gemm is not None:
            gemm.close()
        if float_ops is not None and hasattr(float_ops, "close"):
            float_ops.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="src.runtime.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("package", nargs="?", default=model_env("NPU_COMPILED_PACKAGE"))
    r.add_argument("--package", dest="package_override")
    r.add_argument("--prompt", required=True)
    r.add_argument("--chat", action="store_true")
    r.add_argument("--max-new", "--max-new-tokens", type=int, default=32)
    r.add_argument("--backend", default=None)
    r.add_argument("--overlay", default=os.environ.get("NPU_OVERLAY", "overlay/artifacts/npu_matrix.bit"))
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--top-k", type=int, default=0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--min-p", type=float, default=0.0)
    r.add_argument("--repetition-penalty", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--stream", action="store_true")
    r.add_argument("--json")
    r.set_defaults(fn=cmd_run)
    ref = sub.add_parser("reference", help="Execute the unchanged ONNX operator graph")
    ref.add_argument("--model-dir", default=model_env("NPU_ONNX_MODEL"))
    ref.add_argument("--backend", choices=("cpu", "pynq"), default="cpu")
    ref.add_argument("--overlay", default=os.environ.get("NPU_OVERLAY"))
    ref.add_argument("--prompt", required=True)
    ref.add_argument("--max-new-tokens", type=int, default=24)
    ref.add_argument("--raw", action="store_true")
    ref.set_defaults(fn=cmd_reference)
    args = parser.parse_args(argv)
    if args.cmd == "run" and not (args.package_override or args.package):
        parser.error("Set package, --package or NPU_COMPILED_PACKAGE")
    if args.cmd == "reference" and not args.model_dir:
        parser.error("Set --model-dir or NPU_ONNX_MODEL")
    if args.cmd == "reference" and args.backend == "pynq" and not args.overlay:
        parser.error("The pynq backend requires --overlay or NPU_OVERLAY")
    count = args.max_new if args.cmd == "run" else args.max_new_tokens
    if count < 1:
        parser.error("The generated token limit must be positive")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
