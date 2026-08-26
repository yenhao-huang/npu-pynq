"""Generic runtime boundaries for the PYNQ NPU overlay."""

from .npu import NPURuntime, load_pynq_runtime
from .lowering import (
    LoweringMetrics,
    LoweringResult,
    LoweringValidationError,
    MatrixTileError,
    MatrixLowerer,
)
from .model import (
    LoadedModel,
    ModelExecutionError,
    ModelLoadError,
    ModelMetrics,
    ModelResult,
    ModelRuntimeError,
    NPUModelRuntime,
    load_model_package,
)

__all__ = [
    "NPURuntime",
    "load_pynq_runtime",
    "LoweringMetrics",
    "LoweringResult",
    "LoweringValidationError",
    "MatrixTileError",
    "MatrixLowerer",
    "LoadedModel",
    "ModelExecutionError",
    "ModelLoadError",
    "ModelMetrics",
    "ModelResult",
    "ModelRuntimeError",
    "NPUModelRuntime",
    "load_model_package",
]
