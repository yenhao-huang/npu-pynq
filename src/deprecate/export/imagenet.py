"""ImageNet input preparation and label decoding for the ResNet-18 demo.

The conversion host turns one real image into the exact signed-INT8 tensor the
NPU consumes; the board reloads that tensor and decodes the resulting logits.
Both sides share this module so the preprocessing contract has one definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CROP_SIZE = 224
RESIZE_SHORTER_SIDE = 256
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STANDARD_DEVIATION = (0.229, 0.224, 0.225)
RESAMPLE = "bilinear"
CLASS_COUNT = 1000


class ImageNetInputError(ValueError):
    """The image, preprocessing contract, or class list is unusable."""


def preprocessing_contract() -> dict[str, object]:
    """Return the recorded preprocessing parameters, in serializable form."""

    return {
        "channel_mean": list(CHANNEL_MEAN),
        "channel_standard_deviation": list(CHANNEL_STANDARD_DEVIATION),
        "color_space": "RGB",
        "crop": "center",
        "crop_size": CROP_SIZE,
        "quantization": "symmetric-int8-zero-point-0-round-half-away-from-zero",
        "resample": RESAMPLE,
        "resize_shorter_side": RESIZE_SHORTER_SIDE,
        "version": "torchvision-imagenet1k-v1",
    }


def scale_from_quantization(multiplier_q31: int, shift: int) -> float:
    """Recover the float activation scale from a stored Q1.31 identity."""

    if isinstance(multiplier_q31, bool) or not isinstance(multiplier_q31, int):
        raise ImageNetInputError("multiplier_q31 must be an integer")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ImageNetInputError("shift must be an integer")
    if not 0 < multiplier_q31 < (1 << 31):
        raise ImageNetInputError("multiplier_q31 is outside the Q1.31 range")
    if not 0 <= shift <= 31:
        raise ImageNetInputError("shift is outside the Q1.31 range")
    return float(multiplier_q31) / float(1 << 31) * float(1 << shift)


def input_scale(manifest: Mapping[str, object], tensor_name: str = "input") -> float:
    """Read one tensor's activation scale out of an exported model manifest."""

    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise ImageNetInputError("model manifest has no tensor table")
    for entry in tensors:
        if isinstance(entry, dict) and entry.get("name") == tensor_name:
            quantization = entry.get("quantization")
            if not isinstance(quantization, dict):
                raise ImageNetInputError(f"tensor {tensor_name} has no quantization")
            if quantization.get("zero_point") != 0:
                raise ImageNetInputError("only zero-point-0 tensors are supported")
            return scale_from_quantization(
                quantization.get("multiplier_q31"), quantization.get("shift")
            )
    raise ImageNetInputError(f"model manifest has no tensor {tensor_name}")


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Decode one image file into a full-resolution uint8 RGB HWC array."""

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImageNetInputError(
            "Pillow is required to decode a real image; install pillow"
        ) from error
    try:
        with Image.open(Path(path)) as handle:
            handle.load()
            converted = handle.convert("RGB")
    except OSError as error:
        raise ImageNetInputError(f"cannot decode image {path}: {error}") from error
    array = np.asarray(converted, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ImageNetInputError("decoded image must be a non-empty RGB array")
    return np.ascontiguousarray(array)


def resize_and_center_crop(image: np.ndarray) -> np.ndarray:
    """Resize the shorter side to 256 and center crop 224, as uint8 RGB."""

    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ImageNetInputError("image must be a uint8 RGB HWC array")
    height, width = int(array.shape[0]), int(array.shape[1])
    shorter = min(height, width)
    if shorter < 1:
        raise ImageNetInputError("image must have a positive shorter side")
    ratio = RESIZE_SHORTER_SIDE / shorter
    target = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    resized = Image.fromarray(array, mode="RGB").resize(target, Image.BILINEAR)
    resized_array = np.asarray(resized, dtype=np.uint8)
    resized_height, resized_width = resized_array.shape[:2]
    if resized_height < CROP_SIZE or resized_width < CROP_SIZE:
        raise ImageNetInputError("resized image is smaller than the 224 crop")
    top = (resized_height - CROP_SIZE) // 2
    left = (resized_width - CROP_SIZE) // 2
    crop = resized_array[top : top + CROP_SIZE, left : left + CROP_SIZE, :]
    return np.ascontiguousarray(crop)


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """Convert one uint8 RGB 224 crop into a normalized float32 NCHW tensor."""

    array = np.asarray(crop)
    if array.dtype != np.uint8 or array.shape != (CROP_SIZE, CROP_SIZE, 3):
        raise ImageNetInputError(f"crop must be uint8 ({CROP_SIZE}, {CROP_SIZE}, 3)")
    scaled = array.astype(np.float32) / np.float32(255.0)
    mean = np.array(CHANNEL_MEAN, dtype=np.float32)
    standard_deviation = np.array(CHANNEL_STANDARD_DEVIATION, dtype=np.float32)
    normalized = (scaled - mean) / standard_deviation
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, :, :, :])


def quantize_input(normalized: np.ndarray, scale: float) -> np.ndarray:
    """Quantize a normalized NCHW tensor into the signed-INT8 NHWC NPU input."""

    array = np.asarray(normalized, dtype=np.float32)
    if array.shape != (1, 3, CROP_SIZE, CROP_SIZE):
        raise ImageNetInputError("normalized input must be (1, 3, 224, 224)")
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ImageNetInputError("input scale must be finite and positive")
    ratios = array.astype(np.float64) / float(scale)
    rounded = np.sign(ratios) * np.floor(np.abs(ratios) + 0.5)
    clipped = np.clip(rounded, -127, 127).astype(np.int8)
    return np.ascontiguousarray(clipped.transpose(0, 2, 3, 1))


def preprocess_image(image: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the displayed 224 crop and the signed-INT8 NHWC NPU input."""

    crop = resize_and_center_crop(image)
    return crop, quantize_input(normalize_crop(crop), scale)


def dequantized_preview(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Reverse quantization and normalization to show what the NPU received."""

    array = np.asarray(quantized)
    if array.dtype != np.int8 or array.shape != (1, CROP_SIZE, CROP_SIZE, 3):
        raise ImageNetInputError("quantized input must be int8 (1, 224, 224, 3)")
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ImageNetInputError("input scale must be finite and positive")
    normalized = array.astype(np.float32) * np.float32(scale)
    mean = np.array(CHANNEL_MEAN, dtype=np.float32)
    standard_deviation = np.array(CHANNEL_STANDARD_DEVIATION, dtype=np.float32)
    restored = normalized[0] * standard_deviation + mean
    return np.ascontiguousarray(
        np.clip(restored * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    )


def load_class_names(path: str | Path) -> tuple[str, ...]:
    """Load exactly 1000 newline-separated ImageNet-1K class names."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ImageNetInputError(f"cannot read class names: {error}") from error
    names = tuple(line.strip() for line in text.splitlines() if line.strip())
    if len(names) != CLASS_COUNT:
        raise ImageNetInputError(
            f"class list must hold {CLASS_COUNT} names, found {len(names)}"
        )
    return names


def resolve_class_index(selector: str, names: Sequence[str]) -> int:
    """Resolve an expected class given as an index or an exact class name."""

    text = str(selector).strip()
    if not text:
        raise ImageNetInputError("expected class selector is empty")
    if text.isdigit():
        index = int(text)
        if not 0 <= index < len(names):
            raise ImageNetInputError(f"class index {index} is out of range")
        return index
    matches = [position for position, name in enumerate(names) if name == text]
    if len(matches) != 1:
        raise ImageNetInputError(f"class name {text!r} is not uniquely defined")
    return matches[0]


@dataclass(frozen=True)
class Prediction:
    """One decoded ImageNet class prediction."""

    index: int
    name: str
    logit: float
    probability: float

    def as_record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "logit": round(self.logit, 6),
            "name": self.name,
            "probability": round(self.probability, 6),
        }


def decode_logits(
    logits: np.ndarray,
    scale: float,
    names: Sequence[str],
    *,
    top_k: int = 5,
) -> list[Prediction]:
    """Dequantize INT8 logits and return the highest-scoring classes."""

    array = np.asarray(logits)
    if array.dtype != np.int8:
        raise ImageNetInputError("logits must be signed INT8")
    flat = array.reshape(-1)
    if flat.size != len(names):
        raise ImageNetInputError(
            f"logit count {flat.size} differs from class count {len(names)}"
        )
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ImageNetInputError("logit scale must be finite and positive")
    if not 1 <= int(top_k) <= flat.size:
        raise ImageNetInputError("top_k is outside the available class range")
    values = flat.astype(np.float64) * float(scale)
    shifted = values - values.max()
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    order = np.argsort(-values, kind="stable")[: int(top_k)]
    return [
        Prediction(
            index=int(position),
            name=names[int(position)],
            logit=float(values[int(position)]),
            probability=float(probabilities[int(position)]),
        )
        for position in order
    ]


def canonical_json(value: object) -> bytes:
    """Encode one canonical JSON document, matching the repository contract."""

    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
