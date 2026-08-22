"""Host-testable runtime boundaries for the PYNQ NPU overlay."""

from .matrix_multiplication import (
    MatrixMultiplicationMetrics,
    MatrixMultiplicationResult,
    TiledMatrixMultiplier,
)
from .npu import NPURuntime, load_pynq_runtime

__all__ = [
    "MatrixMultiplicationMetrics",
    "MatrixMultiplicationResult",
    "NPURuntime",
    "TiledMatrixMultiplier",
    "load_pynq_runtime",
]
