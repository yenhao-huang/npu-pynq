#!/usr/bin/env python3
"""Headless entry point for the TPU-Gen POC.

    ./run_flow.py "4x4 INT8 TPU with DRUM_APTPU multiplier and APPROX5 adder"

Same pipeline as demo.ipynb, without a notebook kernel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import paths  # noqa: E402
from flow import MaxIterations, PPAWithinTarget, TPUGenFlow  # noqa: E402
from llm import available_backends, get_backend  # noqa: E402
from openroad import DEFAULT_IMAGE, ORFSConfig  # noqa: E402
from tpugen_types import FlowError, PPATarget  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", help="what to build, in plain English")
    ap.add_argument("--backend", default="replay",
                    choices=available_backends())
    ap.add_argument("--model", default=None,
                    help="model id for the selected backend, if it takes one")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=None,
                    help="stop once PPA is within this fraction of target")
    ap.add_argument("--mode", default="rag", choices=["rag", "full"],
                    help="retrieval mode; 'full' hands OpenROAD the whole library")
    ap.add_argument("--clock-period", type=float, default=5.0, help="ns")
    ap.add_argument("--core-utilization", type=int, default=30)
    ap.add_argument("--openroad-image", default=DEFAULT_IMAGE,
                    help="local OpenROAD/ORFS Docker image (default: %(default)s)")
    ap.add_argument("--target-area", type=float, default=32208.0)
    ap.add_argument("--target-wns", type=float, default=-2.51)
    ap.add_argument("--target-power", type=float, default=0.036)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    target = PPATarget(args.target_area, args.target_wns, args.target_power)
    stop = (
        PPAWithinTarget(target, args.tolerance)
        if args.tolerance is not None
        else MaxIterations(args.iterations)
    )

    backend_kwargs = {"model": args.model} if args.model else {}

    try:
        tpugen = TPUGenFlow(
            get_backend(args.backend, **backend_kwargs),
            orfs=ORFSConfig(
                image=args.openroad_image,
                clock_period=args.clock_period,
                core_utilization=args.core_utilization,
            ),
            retrieval_mode=args.mode,
        )
        result = tpugen.run(
            args.prompt, target, stop=stop,
            max_iterations=args.iterations, run_id=args.run_id,
        )
    except FlowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.table())
    print(f"\nstopped: {result.stopped_because}")
    print(f"run dir: {result.run_dir}")
    if result.best:
        print(f"GDSII  : {result.best.gds}")
        return 0
    print("no iteration reached GDSII", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
