"""Read safetensors checkpoints and ONNX initializers through shared weight access.

A safetensors file is an 8-byte little-endian header length, a JSON header
mapping tensor names to dtype/shape/byte ranges, then the raw data. Tensors
are memory-mapped and converted to float32 on access (bf16 is the upper
half of an fp32).
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import os

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64,
           "I32": np.int32, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


class SafeTensors:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.files: dict[str, tuple[np.memmap, dict]] = {}
        if path.is_dir():
            index = path / "model.safetensors.index.json"
            if index.exists():
                files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
                paths = [path / f for f in files]
            else:
                paths = sorted(path.glob("*.safetensors"))
        else:
            paths = [path]
        if not paths:
            raise FileNotFoundError(f"no safetensors under {path}")
        self.index: dict[str, tuple[Path, dict]] = {}
        self._maps: dict[Path, np.memmap] = {}
        for p in paths:
            with open(p, "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(size))
            header.pop("__metadata__", None)
            self._maps[p] = np.memmap(p, dtype=np.uint8, mode="r", offset=8 + size)
            for name, info in header.items():
                self.index[name] = (p, info)

    def keys(self) -> list[str]:
        return list(self.index)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str) -> np.ndarray:
        path, info = self.index[name]
        start, end = info["data_offsets"]
        raw = self._maps[path][start:end]
        dtype = _DTYPES[info["dtype"]]
        array = np.frombuffer(raw, dtype=dtype).reshape(info["shape"])
        if info["dtype"] == "BF16":
            return (array.astype(np.uint32) << 16).view(np.float32)
        if info["dtype"] == "F16":
            return array.astype(np.float32)
        return np.array(array)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get(name)


class OnnxWeights:
    """Read initializer arrays without changing their dtype or quantization."""
    def __init__(self, path: str | Path):
        import onnx
        from onnx import numpy_helper
        self.model = onnx.load(path)
        self.arrays = {t.name:numpy_helper.to_array(t) for t in self.model.graph.initializer}

    def keys(self):
        return list(self.arrays)

    def __contains__(self, name):
        return name in self.arrays

    def get(self, name):
        return self.arrays[name]

    def __getitem__(self, name):
        return self.get(name)


def load_weights(path: str | Path):
    """Use the same initializer interface for ONNX and safetensors sources."""
    path = Path(path)
    return OnnxWeights(path) if path.suffix == '.onnx' else SafeTensors(path)


def onnx_asset_dir(root: str | Path, name: str) -> Path:
    """Resolve current asset names, accepting pre-rename local model stores."""
    root = Path(root)
    current = root / name
    legacy = {'onnx':'official-onnx', 'onnx-graph':'official-graph',
              'onnx-ops':'official-ort-ops', 'onnx-runtime':'official-ort'}
    previous = root / legacy.get(name, name)
    return previous if not current.exists() and previous.exists() else current


def model_env(name: str):
    """Read canonical model settings, with old notebook settings as fallback."""
    legacy = {'NPU_ONNX_MODEL':'NPU_OFFICIAL_MODEL',
              'NPU_COMPILED_PACKAGE':'NPU_OFFICIAL_COMPILED'}
    return os.environ.get(name, os.environ.get(legacy.get(name, name)))


def resolve_hf(model_id_or_path: str) -> Path:
    """A local directory, or the newest snapshot of a model in the HF cache."""
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path
    import os
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshots = cache / f"models--{model_id_or_path.replace('/', '--')}" / "snapshots"
    candidates = sorted(snapshots.glob("*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"{model_id_or_path} is neither a path nor in {cache}")
    return candidates[-1]
