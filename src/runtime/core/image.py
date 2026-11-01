"""Image classification through an already compiled package (no compiler imports)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


# Image classification runs an npu-package/2 package through the delegate runtime.
from .delegated import DelegatedImageEngine as ImageEngine  # noqa: E402
