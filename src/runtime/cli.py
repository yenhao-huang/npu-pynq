"""Run an exported LLM package on the board (or the host simulator).

    python -m src.runtime.cli run package --overlay npu_matrix.bit --chat --stream --prompt "Hello"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def cmd_run(args) -> int:
    from src.runtime.core.delegated import ExportedLLM, on_board
    from src.runtime.core.sampling import SamplingParams

    overlay = None
    if (args.backend or ('pynq' if on_board() else 'sim')) == 'pynq':
        from pynq import Overlay
        overlay = Overlay(args.overlay)
    llm = ExportedLLM(Path(args.package), overlay)
    params = SamplingParams(args.temperature, args.top_k, args.top_p, args.min_p, args.repetition_penalty, args.seed)

    def show(_tid, piece):
        sys.stdout.write(piece)
        sys.stdout.flush()

    ids, text, stats = llm.generate(args.prompt, args.max_new, params, chat=args.chat,
                                    on_token=show if args.stream else None)
    if not args.stream:
        print(text)
    print()
    report = {"prompt": args.prompt, "output": text, "output_ids": ids, **stats}
    print(json.dumps(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    llm.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="src.runtime.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Generate with an exported LLM package")
    r.add_argument("package", nargs="?", default=os.environ.get("NPU_COMPILED_PACKAGE"))
    r.add_argument("--prompt", required=True)
    r.add_argument("--chat", action="store_true")
    r.add_argument("--max-new", "--max-new-tokens", type=int, default=32)
    r.add_argument("--backend", choices=("sim", "pynq"), default=None)
    r.add_argument("--overlay", default=os.environ.get("NPU_OVERLAY", "npu_matrix.bit"))
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
    if not args.package:
        parser.error("Set package or NPU_COMPILED_PACKAGE")
    if args.max_new < 1:
        parser.error("The generated token limit must be positive")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
