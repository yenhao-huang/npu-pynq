"""Non-interactive Phase 1C acceptance runner for a packaged PYNQ overlay.

It tiles logical M/N matrix multiplications over the legacy DMA runtime and
checks them against NumPy. The tiling lives here, not in src/runtime/, because
only this acceptance and the matrix example use it; the example notebook keeps
its own copy (examples may not ship .py files).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


from src.runtime import load_pynq_runtime
from src.vendors.npu_pynq.driver.overlay.verify import verify_artifacts


PASS_MARKER = "PASS: Phase 1C matrix multiplication example"


@dataclass(frozen=True)
class MatrixMultiplicationMetrics:
    """Measured accounting for one complete logical multiplication."""

    m: int
    n: int
    k: int
    tile_count: int
    elapsed_seconds: float
    mac_count: int
    operation_count: int
    operations_per_second: float


@dataclass(frozen=True)
class MatrixMultiplicationResult:
    """Owned signed INT32 output and its immutable accounting metadata."""

    output: np.ndarray
    metrics: MatrixMultiplicationMetrics


class TiledMatrixMultiplier:
    """Tile logical M/N dimensions over one physical ``NPURuntime`` instance."""

    def __init__(
        self,
        runtime: Any,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.monotonic = monotonic
        try:
            self.max_m = int(runtime.max_m)
            self.max_n = int(runtime.max_n)
            self.max_k = int(runtime.max_k)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("runtime physical limits are missing") from error
        if self.max_m <= 0 or self.max_n <= 0 or self.max_k <= 0:
            raise ValueError("runtime physical limits must be positive")
        if not callable(getattr(runtime, "run", None)):
            raise TypeError("runtime must expose a callable run method")

    def run(
        self,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        *,
        hardware_timeout_cycles: int = 1_000_000,
        software_timeout: float = 5.0,
    ) -> MatrixMultiplicationResult:
        m, n, k, timeout = self._validate(
            a_matrix,
            b_matrix,
            hardware_timeout_cycles,
            software_timeout,
        )
        start = float(self.monotonic())
        deadline = start + timeout
        output = np.empty((m, n), dtype=np.int32, order="C")
        tile_count = 0

        for row_start in range(0, m, self.max_m):
            row_stop = min(row_start + self.max_m, m)
            for column_start in range(0, n, self.max_n):
                column_stop = min(column_start + self.max_n, n)
                remaining = deadline - float(self.monotonic())
                if remaining <= 0.0:
                    raise TimeoutError("logical matrix multiplication timed out")
                a_tile = np.ascontiguousarray(
                    a_matrix[row_start:row_stop, :], dtype=np.int8
                )
                b_tile = np.ascontiguousarray(
                    b_matrix[:, column_start:column_stop], dtype=np.int8
                )
                tile_output = np.asarray(
                    self.runtime.run(
                        a_tile,
                        b_tile,
                        hardware_timeout_cycles=hardware_timeout_cycles,
                        software_timeout=remaining,
                    )
                )
                expected_shape = (row_stop - row_start, column_stop - column_start)
                if tile_output.dtype != np.int32 or tile_output.shape != expected_shape:
                    raise RuntimeError(
                        "physical runtime returned an incompatible output: "
                        f"dtype={tile_output.dtype} shape={tile_output.shape}, "
                        f"expected int32 {expected_shape}"
                    )
                output[row_start:row_stop, column_start:column_stop] = tile_output
                tile_count += 1

        elapsed = float(self.monotonic()) - start
        if elapsed < 0.0:
            raise RuntimeError("monotonic clock moved backwards")
        mac_count = m * n * k
        operation_count = 2 * mac_count
        throughput = (
            operation_count / elapsed if elapsed > 0.0 else float("inf")
        )
        metrics = MatrixMultiplicationMetrics(
            m=m,
            n=n,
            k=k,
            tile_count=tile_count,
            elapsed_seconds=elapsed,
            mac_count=mac_count,
            operation_count=operation_count,
            operations_per_second=throughput,
        )
        return MatrixMultiplicationResult(output=np.array(output, copy=True), metrics=metrics)

    def _validate(
        self,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        hardware_timeout_cycles: int,
        software_timeout: float,
    ) -> tuple[int, int, int, float]:
        if not isinstance(a_matrix, np.ndarray) or not isinstance(b_matrix, np.ndarray):
            raise TypeError("A and B must be NumPy arrays")
        if a_matrix.dtype != np.int8 or b_matrix.dtype != np.int8:
            raise ValueError("A and B must use signed INT8 dtype")
        if a_matrix.ndim != 2 or b_matrix.ndim != 2:
            raise ValueError("A and B must be rank-two matrices")
        m, k = map(int, a_matrix.shape)
        b_k, n = map(int, b_matrix.shape)
        if m <= 0 or n <= 0 or k <= 0 or b_k <= 0:
            raise ValueError("matrix dimensions must be positive")
        if k != b_k:
            raise ValueError("A columns must equal B rows")
        if k > self.max_k:
            raise ValueError(
                f"K={k} exceeds the physical MAX_K={self.max_k}; exact K tiling is unsupported"
            )
        if (
            isinstance(hardware_timeout_cycles, bool)
            or not isinstance(hardware_timeout_cycles, int)
            or hardware_timeout_cycles <= 0
        ):
            raise ValueError("hardware timeout must be a positive integer")
        if isinstance(software_timeout, bool) or not isinstance(software_timeout, Real):
            raise TypeError("software timeout must be a positive finite number")
        timeout = float(software_timeout)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("software timeout must be a positive finite number")
        return m, n, k, timeout


class BoardExampleError(RuntimeError):
    """The packaged Phase 1C example did not satisfy board acceptance."""


def _reference(a_matrix: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
    return (a_matrix.astype(np.int64) @ b_matrix.astype(np.int64)).astype(np.int32)


def _case_record(name: str, result: Any) -> dict[str, object]:
    metrics = result.metrics
    throughput = float(metrics.operations_per_second)
    return {
        "name": name,
        "status": "PASS",
        "shape": [metrics.m, metrics.k, metrics.n],
        "tile_count": metrics.tile_count,
        "elapsed_seconds": metrics.elapsed_seconds,
        "operation_count": metrics.operation_count,
        "operations_per_second": throughput if math.isfinite(throughput) else None,
    }


def execute_cases(
    runtime: Any,
    manifest: dict[str, object],
    *,
    release_tag: str,
) -> dict[str, object]:
    """Execute required cases through public runtimes and return PASS evidence."""

    multiplier = TiledMatrixMultiplier(runtime)
    normal_a = ((np.arange(runtime.max_m * 5).reshape(runtime.max_m, 5) % 17) - 8).astype(np.int8)
    normal_b = ((np.arange(5 * runtime.max_n).reshape(5, runtime.max_n) % 19) - 9).astype(np.int8)
    non_aligned_a = ((np.arange((runtime.max_m + 1) * 5).reshape(runtime.max_m + 1, 5) % 17) - 8).astype(np.int8)
    non_aligned_b = ((np.arange(5 * (runtime.max_n + 1)).reshape(5, runtime.max_n + 1) % 19) - 9).astype(np.int8)
    cases = (
        ("normal", normal_a, normal_b),
        ("non_aligned", non_aligned_a, non_aligned_b),
        ("repeated", -non_aligned_a, non_aligned_b),
    )

    records: list[dict[str, object]] = []
    for name, a_matrix, b_matrix in cases:
        try:
            result = multiplier.run(a_matrix, b_matrix, software_timeout=10.0)
            np.testing.assert_array_equal(result.output, _reference(a_matrix, b_matrix))
        except Exception as error:
            raise BoardExampleError(f"{name} matrix case failed") from error
        records.append(_case_record(name, result))

    if int(records[1]["tile_count"]) <= 1:
        raise BoardExampleError("non-aligned case did not exercise physical tiling")

    return {
        "schema_version": 1,
        "release_tag": release_tag,
        "source_commit": manifest.get("source_commit", "unknown"),
        "vivado_version": manifest.get("vivado_version", "unknown"),
        "target_part": manifest.get("target_part", "unknown"),
        "bit": manifest.get("bit", {}),
        "hwh": manifest.get("hwh", {}),
        "physical_limits": [runtime.max_m, runtime.max_n, runtime.max_k],
        "cases": records,
        "pass_marker": PASS_MARKER,
    }


def _write_evidence(path: Path, evidence: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    arguments = parser.parse_args()
    try:
        artifact_dir = arguments.artifact_dir.resolve()
        manifest = verify_artifacts(artifact_dir)
        runtime = load_pynq_runtime(artifact_dir / "npu_matrix.bit")
        evidence = execute_cases(
            runtime,
            manifest,
            release_tag=arguments.release_tag,
        )
        _write_evidence(arguments.evidence, evidence)
    except Exception as error:
        print(f"Phase 1C board example failed: {error}", file=sys.stderr)
        return 1
    print(PASS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
