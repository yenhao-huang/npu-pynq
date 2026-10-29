"""The load stage: find a model source and read it without changing any
number. Safetensors checkpoints, ONNX initializers and the pinned ONNX
preparation live here; detect_format decides which one a path is, and is
the only place that does."""

from __future__ import annotations

from pathlib import Path

from src.export.load.onnx import OnnxWeights
from src.export.load.safetensors import SafeTensors
from src.runtime.core.package import onnx_asset_dir

FORMATS = ('safetensors', 'onnx', 'torch')


def is_onnx(path: str | Path) -> bool:
    """An .onnx file, or a prepared directory holding model_int8.onnx."""
    path = Path(path).expanduser()
    return (path.suffix == '.onnx' or (path / 'model_int8.onnx').is_file()
            or (onnx_asset_dir(path, 'onnx') / 'model_int8.onnx').is_file())


def detect_format(source: str | Path | None, requested: str = 'auto') -> str:
    """``requested`` unless it is 'auto'; otherwise the format ``source`` holds."""
    if requested != 'auto':
        return requested
    if source is None:
        return 'safetensors'
    path = Path(source).expanduser()
    if is_onnx(path):
        return 'onnx'
    if path.suffix in ('.pth', '.pt'):
        return 'torch'
    return 'safetensors'


def load_weights(path: str | Path):
    """Use the same initializer interface for ONNX and safetensors sources."""
    path = Path(path)
    return OnnxWeights(path) if path.suffix == '.onnx' else SafeTensors(path)
