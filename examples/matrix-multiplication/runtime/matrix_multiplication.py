"""Logical M/N-tiled matrix multiplication over the bounded NPU runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import time
from typing import Any, Callable

import numpy as np


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
