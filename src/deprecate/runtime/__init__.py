"""Pre-MLIR ResNet runtime and acceptance API."""

from src.runtime.npu import NPURuntime, PhysicalJobMetrics, load_pynq_runtime
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
from .acceptance import AcceptanceRunError, run_resnet18_acceptance

__all__ = [
    "NPURuntime",
    "PhysicalJobMetrics",
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
    "AcceptanceRunError",
    "run_resnet18_acceptance",
]
