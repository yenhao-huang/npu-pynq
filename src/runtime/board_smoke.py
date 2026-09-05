"""Physical-board smoke test for the Phase 1B DMA matrix vertical slice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .npu import (
    REG_CYCLES_HI,
    REG_CYCLES_LO,
    REG_ERROR,
    REG_STATUS,
    STATUS_DONE,
    load_pynq_runtime,
)
from .verify_overlay import verify_artifacts


PASS_MARKER = "PASS: NPU DMA matrix vertical slice"


def execute_smoke(runtime: Any, manifest: dict[str, object]) -> dict[str, object]:
    matrix_a = np.array([[-128, 127], [7, -3]], dtype=np.int8)
    matrix_b = np.array([[-1, 2], [4, -5]], dtype=np.int8)
    expected = np.array([[127, -128], [-19, 29]], dtype=np.int8)
    actual = runtime.run(
        matrix_a,
        matrix_b,
        bias=np.zeros((2,), dtype=np.int32),
        multipliers_q31=np.full((2,), (1 << 31) - 1, dtype=np.int32),
        shifts=np.zeros((2,), dtype=np.uint8),
        output_zero_point=0,
        hardware_timeout_cycles=1_000_000,
        software_timeout=10.0,
    )
    if not np.array_equal(actual, expected):
        raise RuntimeError(f"board result mismatch: expected {expected.tolist()}, got {actual.tolist()}")
    status = int(runtime.mmio.read(REG_STATUS))
    error = int(runtime.mmio.read(REG_ERROR))
    cycles = int(runtime.mmio.read(REG_CYCLES_LO)) | (
        int(runtime.mmio.read(REG_CYCLES_HI)) << 32
    )
    cycles_again = int(runtime.mmio.read(REG_CYCLES_LO)) | (
        int(runtime.mmio.read(REG_CYCLES_HI)) << 32
    )
    if status != STATUS_DONE or error != 0 or cycles <= 0 or cycles_again != cycles:
        raise RuntimeError(
            f"board status mismatch: status={status} error={error} cycles={cycles}/{cycles_again}"
        )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest.get("source_commit", "unknown"),
        "vivado_version": manifest.get("vivado_version", "unknown"),
        "target_part": manifest.get("target_part", "unknown"),
        "bit": manifest.get("bit", {}),
        "hwh": manifest.get("hwh", {}),
        "metadata": manifest.get("metadata", {}),
        "abi": {"status": status, "error": error, "cycles": cycles},
        "result": actual.tolist(),
        "pass_marker": PASS_MARKER,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        artifact_dir = arguments.artifact_dir.resolve()
        manifest = verify_artifacts(artifact_dir)
        runtime = load_pynq_runtime(artifact_dir / "npu_matrix.bit")
        evidence = execute_smoke(runtime, manifest)
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as error:
        print(f"NPU board smoke failed: {error}", file=sys.stderr)
        return 1
    print(PASS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
