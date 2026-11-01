"""Image classification through an already compiled package (no compiler imports)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np

from .package import Package, platform_backend


def preprocess(image, contract: dict) -> np.ndarray:
    """Apply the exported RGB/resize/center-crop/normalization contract."""
    from PIL import Image

    image = image.convert("RGB")
    h, w, channels = contract["shape"]
    if channels != 3 or contract.get("layout") != "NHWC":
        raise ValueError("Image classification requires RGB NHWC input")
    resize = contract.get("resize_shorter", 256)
    iw, ih = image.size
    scale = resize / min(iw, ih)
    image = image.resize((int(iw * scale), int(ih * scale)),
                         getattr(Image, 'Resampling', Image).BILINEAR)
    iw, ih = image.size
    left, top = int(round((iw - w) / 2)), int(round((ih - h) / 2))
    pixels = np.asarray(image.crop((left, top, left + w, top + h)), np.float32) / 255
    return np.ascontiguousarray((pixels - np.asarray(contract["mean"], np.float32)) /
                                np.asarray(contract["std"], np.float32))


@dataclass(frozen=True)
class Prediction:
    index: int
    name: str
    probability: float


def top_predictions(logits, labels, k=5):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(logits) != len(labels) or not np.isfinite(logits).all():
        raise ValueError("Finite logits and one label per class are required")
    if not 1 <= k <= len(labels):
        raise ValueError("Invalid top-k")
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return [Prediction(int(i), labels[i], float(probabilities[i]))
            for i in np.argsort(-probabilities, kind="stable")[:k]]


class ImageEngine:
    def __new__(cls, package, *args, **kwargs):
        # Packages exported through PyTorch run through the delegate runtime.
        manifest = Path(package) / "manifest.json"
        if cls is ImageEngine and manifest.exists() and                 json.loads(manifest.read_text()).get("format") == "npu-package/2":
            from .delegated import DelegatedImageEngine
            return DelegatedImageEngine(package, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, package, backend=None, *, overlay=None, stream_weights=None):
        self.package = Package(package, backend or platform_backend(), overlay=overlay,
                               stream_weights=stream_weights)
        meta = self.package.manifest["meta"]
        self.contract = meta["input"]
        self.classes = int(meta.get("classes", 1000))
        if self.package.manifest["entry_points"].get("forward") != ["weights", "image", "logits"]:
            raise ValueError("Unsupported image entry point ABI")
        self.last_seconds = 0.0
        self.last_stats = {}

    def predict(self, image):
        image = np.asarray(image)
        if image.dtype != np.float32 or list(image.shape) != self.contract["shape"]:
            raise ValueError("Expected a normalized float32 image matching the package input")
        if not np.isfinite(image).all():
            raise ValueError("Image contains nonfinite values")
        logits = np.zeros((1, self.classes), np.float32)
        before = self.package.stats()
        start = time.perf_counter()
        self.package.call("forward", "weights", np.ascontiguousarray(image), logits)
        self.last_seconds = time.perf_counter() - start
        after = self.package.stats()
        self.last_stats = {key: after[key] - before[key] for key in
                           ("calls", "jobs", "cycles", "macs", "polls")}
        if not np.isfinite(logits).all():
            raise RuntimeError("Compiled image model returned nonfinite logits")
        return logits[0]
