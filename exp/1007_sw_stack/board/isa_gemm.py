"""Instruction-stream GEMM acceptance on the PYNQ-Z1 (issue #120)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pynq import Overlay, allocate  # noqa: E402

from src.runtime.isa_runtime import IsaDevice, gemm  # noqa: E402
from src.runtime.npu import NPURuntime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(119)
    report = {"cases": [], "pass": True}
    overlay = Overlay(args.overlay)
    acc = overlay.npu_accelerator_0
    report["version"] = hex(int(acc.read(0x04)))
    report["capabilities"] = hex(int(acc.read(0x08)))

    legacy = NPURuntime(overlay, allocator=allocate)
    a = rng.integers(-128, 128, (16, 256), dtype=np.int8)
    b = rng.integers(-128, 128, (256, 16), dtype=np.int8)
    t = time.monotonic()
    c = legacy.run(a, b)
    report["legacy_tile"] = {
        "match": bool(np.array_equal(c, a.astype(np.int32) @ b.astype(np.int32))),
        "seconds": time.monotonic() - t,
    }
    report["pass"] &= report["legacy_tile"]["match"]

    device = IsaDevice(overlay)
    for m, n, k, tk in [(16, 16, 256, None), (5, 37, 300, None), (1, 576, 576, None),
                        (64, 1536, 576, None), (1, 49152, 576, None), (197, 64, 147, None)]:
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        t = time.monotonic()
        c, stats = gemm(device, allocate, a, b, tk)
        wall = time.monotonic() - t
        ok = bool(np.array_equal(c, a.astype(np.int32) @ b.astype(np.int32)))
        report["pass"] &= ok
        case = {"m": m, "n": n, "k": k, "match": ok, "jobs": stats.jobs, "cycles": stats.cycles,
                "instructions": stats.instructions, "npu_seconds": stats.seconds, "wall_seconds": wall,
                "gmacs": m * n * k / max(stats.seconds, 1e-9) / 1e9}
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    # The DMA path still works after the front end has owned the controller.
    a = rng.integers(-128, 128, (16, 256), dtype=np.int8)
    b = rng.integers(-128, 128, (256, 16), dtype=np.int8)
    report["legacy_after_isa"] = bool(np.array_equal(legacy.run(a, b), a.astype(np.int32) @ b.astype(np.int32)))
    report["pass"] &= report["legacy_after_isa"]
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
