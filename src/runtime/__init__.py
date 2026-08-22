"""Generic runtime boundaries for the PYNQ NPU overlay."""

from .npu import NPURuntime, load_pynq_runtime

__all__ = [
    "NPURuntime",
    "load_pynq_runtime",
]
