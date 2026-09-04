"""Pinned TorchVision ResNet-18 to Phase 2A conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from src.export.resnet import ExportedPackage, export_model
from src.model.numeric import INT32_MAX, INT32_MIN
from src.model.resnet import (
    ConstantTensor,
    Conv2D,
    Flatten,
    FullyConnected,
    GlobalAveragePool,
    MaxPool,
    Quantization,
    QuantizedGraph,
    Relu,
    ResidualAdd,
    TensorSpec,
)


FIXTURE_EVIDENCE_TYPE = "software-fixture"
REAL_MODEL_HOST_EVIDENCE_TYPE = "real-model-host"
PHYSICAL_BOARD_EVIDENCE_TYPE = "physical-pynq-z1"


def expected_source_shapes() -> dict[str, tuple[int, ...]]:
    """Return the exact TorchVision ResNet-18 IMAGENET1K_V1 state schema."""

    shapes: dict[str, tuple[int, ...]] = {"conv1.weight": (64, 3, 7, 7)}
    batch_norms: list[tuple[str, int]] = [("bn1", 64)]
    channels = (64, 128, 256, 512)
    input_channels = 64
    for stage, output_channels in enumerate(channels, start=1):
        for block in range(2):
            prefix = f"layer{stage}.{block}"
            block_input = input_channels if block == 0 else output_channels
            shapes[f"{prefix}.conv1.weight"] = (
                output_channels,
                block_input,
                3,
                3,
            )
            shapes[f"{prefix}.conv2.weight"] = (
                output_channels,
                output_channels,
                3,
                3,
            )
            batch_norms.extend(
                ((f"{prefix}.bn1", output_channels), (f"{prefix}.bn2", output_channels))
            )
            if stage > 1 and block == 0:
                shapes[f"{prefix}.downsample.0.weight"] = (
                    output_channels,
                    block_input,
                    1,
                    1,
                )
                batch_norms.append((f"{prefix}.downsample.1", output_channels))
        input_channels = output_channels
    for prefix, count in batch_norms:
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            shapes[f"{prefix}.{suffix}"] = (count,)
    shapes["fc.weight"] = (1000, 512)
    shapes["fc.bias"] = (1000,)
    return dict(sorted(shapes.items()))


def validate_source_arrays(state: Mapping[str, np.ndarray]) -> None:
    """Reject any source state that differs from the pinned tensor schema."""

    if not isinstance(state, Mapping):
        raise TypeError("state must be a tensor mapping")
    expected = expected_source_shapes()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        names = missing or extra
        kind = "missing" if missing else "unexpected"
        raise ValueError(f"{kind} source tensor: {names[0]}")
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"source tensor {name} must be a NumPy array")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"source tensor {name} shape {tuple(value.shape)} != {shape}"
            )
        if value.dtype.kind != "f":
            raise ValueError(f"source tensor {name} must be floating point")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"source tensor {name} contains non-finite values")


def _finite_float_array(name: str, value: np.ndarray, rank: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank or array.dtype.kind != "f":
        raise ValueError(f"{name} must be a rank-{rank} floating-point array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float64)


def fold_batch_norm(
    weight: np.ndarray,
    bias: np.ndarray | None,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_variance: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold one inference BatchNorm into an OIHW convolution."""

    weights = _finite_float_array("weight", weight, 4)
    channels = int(weights.shape[0])
    vectors = []
    for name, value in (
        ("gamma", gamma),
        ("beta", beta),
        ("running_mean", running_mean),
        ("running_variance", running_variance),
    ):
        vector = _finite_float_array(name, value, 1)
        if vector.shape != (channels,):
            raise ValueError(f"{name} shape must be ({channels},)")
        vectors.append(vector)
    if bias is None:
        biases = np.zeros((channels,), dtype=np.float64)
    else:
        biases = _finite_float_array("bias", bias, 1)
        if biases.shape != (channels,):
            raise ValueError(f"bias shape must be ({channels},)")
    if not isinstance(epsilon, (int, float)) or not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    gamma_value, beta_value, mean_value, variance_value = vectors
    denominator = variance_value + float(epsilon)
    if np.any(denominator <= 0):
        raise ValueError("BatchNorm variance plus epsilon must be positive")
    factor = gamma_value / np.sqrt(denominator)
    folded_weight = weights * factor[:, None, None, None]
    folded_bias = beta_value + (biases - mean_value) * factor
    return (
        np.ascontiguousarray(folded_weight, dtype=np.float32),
        np.ascontiguousarray(folded_bias, dtype=np.float32),
    )


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    magnitudes = np.floor(np.abs(values) + 0.5)
    return np.copysign(magnitudes, values)


def quantize_conv_weight(
    weight: np.ndarray, *, minimum_scales: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize OIHW float weights per output channel and return HWIO INT8."""

    weights = _finite_float_array("weight", weight, 4)
    maxima = np.max(np.abs(weights), axis=(1, 2, 3))
    scales = np.where(maxima > 0.0, maxima / 127.0, 1.0)
    if minimum_scales is not None:
        minimum = np.asarray(minimum_scales, dtype=np.float64)
        if (
            minimum.shape != scales.shape
            or not np.all(np.isfinite(minimum))
            or np.any(minimum <= 0.0)
        ):
            raise ValueError("minimum weight scales must be finite, positive, and per-channel")
        scales = np.maximum(scales, minimum)
    quantized = _round_away_from_zero(
        weights / scales[:, None, None, None]
    )
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return (
        np.ascontiguousarray(np.transpose(quantized, (2, 3, 1, 0))),
        np.ascontiguousarray(scales, dtype=np.float64),
    )


def encode_q31_ratio(ratio: float) -> tuple[int, int]:
    """Encode one positive scale ratio in the existing Q1.31 contract."""

    if not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)):
        raise TypeError("ratio must be a finite number")
    ratio = float(ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("Q1.31 scale ratio must be in [0, 1)")
    multiplier = int(math.floor(ratio * (1 << 31) + 0.5))
    if multiplier >= (1 << 31):
        raise ValueError("Q1.31 scale ratio rounds outside signed INT32")
    return multiplier, 0


def activation_quantization(maximum_absolute_value: float) -> Quantization:
    """Return a zero-centered signed-INT8 identity for one observed range."""

    if not isinstance(maximum_absolute_value, (int, float)) or not math.isfinite(
        float(maximum_absolute_value)
    ):
        raise TypeError("activation maximum must be finite")
    maximum = float(maximum_absolute_value)
    if maximum <= 0.0:
        raise ValueError("activation maximum must be positive")
    scale = maximum / 127.0
    if scale >= 1.0:
        multiplier = (1 << 31) - 1
        shift = min(31, int(math.ceil(math.log2(scale))))
    else:
        multiplier, shift = encode_q31_ratio(scale)
    return Quantization(multiplier, shift, 0)


def residual_scale_groups(
    observed_maxima: Mapping[str, float], *, stages: int = 4
) -> dict[str, float]:
    """Assign one symmetric scale to every externally visible stage residual."""

    if not 1 <= stages <= 4:
        raise ValueError("stages must be in [1, 4]")
    result: dict[str, float] = {}
    for stage in range(1, stages + 1):
        prefixes = ("stem.", "layer1.") if stage == 1 else (f"layer{stage}.",)
        names = [
            name
            for name in observed_maxima
            if name.startswith(prefixes)
            and not name.endswith(("conv1", "relu1"))
        ]
        if not names:
            raise ValueError(f"stage {stage} has no residual calibration records")
        maximum = max(float(observed_maxima[name]) for name in names)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(f"stage {stage} calibration maximum is invalid")
        scale = maximum / 127.0
        for name in names:
            result[name] = scale
    return result


def generate_calibration_inputs() -> np.ndarray:
    """Generate deterministic normalized NCHW full-shape calibration tensors."""

    coordinate = np.linspace(0.0, 1.0, 224, dtype=np.float32)
    vertical, horizontal = np.meshgrid(coordinate, coordinate, indexing="ij")
    first = np.stack((horizontal, vertical, (horizontal + vertical) * 0.5))
    checker = ((np.indices((224, 224)).sum(axis=0) // 16) % 2).astype(np.float32)
    second = np.stack((checker, 1.0 - checker, horizontal * vertical))
    images = np.stack((first, second)).astype(np.float32)
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)[None, :, None, None]
    standard_deviation = np.array(
        (0.229, 0.224, 0.225), dtype=np.float32
    )[None, :, None, None]
    return np.ascontiguousarray((images - mean) / standard_deviation)


def compare_integer_captures(
    expected: Mapping[str, np.ndarray], actual: Mapping[str, np.ndarray]
) -> None:
    """Raise at the first stable tensor/index mismatch."""

    if set(expected) != set(actual):
        raise ValueError(
            "capture names differ: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
    for name in sorted(expected):
        reference = np.asarray(expected[name])
        observed = np.asarray(actual[name])
        if reference.dtype != np.int8 or observed.dtype != np.int8:
            raise ValueError(f"capture {name} must use signed INT8")
        if reference.shape != observed.shape:
            raise ValueError(
                f"capture {name} shape {observed.shape} != {reference.shape}"
            )
        differences = np.argwhere(reference != observed)
        if differences.size:
            index = tuple(int(value) for value in differences[0])
            formatted = "[" + ", ".join(str(value) for value in index) + "]"
            raise ValueError(
                f"capture {name} differs at {formatted}: "
                f"expected {int(reference[index])}, got {int(observed[index])}"
            )


@dataclass(frozen=True)
class FloatTrace:
    maxima: Mapping[str, float]
    captures: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ConversionResult:
    package: ExportedPackage
    provenance_path: Path
    input_path: Path
    graph: QuantizedGraph
    input_scale: float


@dataclass(frozen=True)
class _ConvSpec:
    command_id: str
    source_prefix: str
    batch_norm_prefix: str
    input_id: str
    output_id: str
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]


def _conv_specs() -> tuple[_ConvSpec, ...]:
    specs = [
        _ConvSpec(
            "stem.conv",
            "conv1",
            "bn1",
            "input",
            "stem.conv",
            (2, 2),
            (3, 3, 3, 3),
        )
    ]
    prior = "stem.pool"
    for stage in range(1, 5):
        for block in range(2):
            prefix = f"layer{stage}.{block}"
            stride = (2, 2) if stage > 1 and block == 0 else (1, 1)
            specs.append(
                _ConvSpec(
                    f"{prefix}.conv1",
                    f"{prefix}.conv1",
                    f"{prefix}.bn1",
                    prior,
                    f"{prefix}.conv1",
                    stride,
                    (1, 1, 1, 1),
                )
            )
            specs.append(
                _ConvSpec(
                    f"{prefix}.conv2",
                    f"{prefix}.conv2",
                    f"{prefix}.bn2",
                    f"{prefix}.relu1",
                    f"{prefix}.conv2",
                    (1, 1),
                    (1, 1, 1, 1),
                )
            )
            if stage > 1 and block == 0:
                specs.append(
                    _ConvSpec(
                        f"{prefix}.projection",
                        f"{prefix}.downsample.0",
                        f"{prefix}.downsample.1",
                        prior,
                        f"{prefix}.projection",
                        stride,
                        (0, 0, 0, 0),
                    )
                )
            prior = f"{prefix}.relu"
    return tuple(specs)


def load_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Safely load only tensors from the pinned TorchVision state dictionary."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required only for conversion; install requirements-convert.txt"
        ) from error
    try:
        loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"checkpoint cannot be loaded safely: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint must contain one state dictionary")
    state: dict[str, np.ndarray] = {}
    for name, value in loaded.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint must contain only named tensors")
        if value.layout != torch.strided or value.device.type != "cpu":
            raise ValueError(f"source tensor {name} must use dense CPU storage")
        state[name] = value.detach().numpy()
    validate_source_arrays(state)
    return state


def _folded_convolutions(
    state: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for spec in _conv_specs():
        conv = spec.source_prefix
        batch_norm = spec.batch_norm_prefix
        result[spec.command_id] = fold_batch_norm(
            state[f"{conv}.weight"],
            None,
            state[f"{batch_norm}.weight"],
            state[f"{batch_norm}.bias"],
            state[f"{batch_norm}.running_mean"],
            state[f"{batch_norm}.running_var"],
        )
    return result


def run_float_resnet18(
    state: Mapping[str, np.ndarray], inputs: np.ndarray
) -> FloatTrace:
    """Interpret the pinned architecture and capture stable NCHW float tensors."""

    validate_source_arrays(state)
    values = np.asarray(inputs)
    if values.dtype != np.float32 or values.ndim != 4 or values.shape[1:] != (3, 224, 224):
        raise ValueError("float inputs must have shape (N, 3, 224, 224) and float32 dtype")
    if values.shape[0] <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("float inputs must be finite and non-empty")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("PyTorch is required for float conversion calibration") from error
    folded = _folded_convolutions(state)
    captures: dict[str, np.ndarray] = {}
    maxima: dict[str, float] = {"input": float(np.max(np.abs(values)))}

    def record(name, tensor):
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        captures[name] = np.ascontiguousarray(array)
        maxima[name] = float(np.max(np.abs(array)))
        return tensor

    def conv(name, tensor, stride, padding):
        weights, biases = folded[name]
        return functional.conv2d(
            tensor,
            torch.from_numpy(weights),
            torch.from_numpy(biases),
            stride=stride,
            padding=(padding[0], padding[2]),
        )

    with torch.no_grad():
        current = torch.from_numpy(np.ascontiguousarray(values))
        current = record("stem.conv", conv("stem.conv", current, (2, 2), (3, 3, 3, 3)))
        current = record("stem.relu", functional.relu(current))
        current = record(
            "stem.pool",
            functional.max_pool2d(current, kernel_size=3, stride=2, padding=1),
        )
        for stage in range(1, 5):
            for block in range(2):
                prefix = f"layer{stage}.{block}"
                identity = current
                stride = (2, 2) if stage > 1 and block == 0 else (1, 1)
                branch = record(
                    f"{prefix}.conv1",
                    conv(f"{prefix}.conv1", current, stride, (1, 1, 1, 1)),
                )
                branch = record(f"{prefix}.relu1", functional.relu(branch))
                branch = record(
                    f"{prefix}.conv2",
                    conv(f"{prefix}.conv2", branch, (1, 1), (1, 1, 1, 1)),
                )
                if stage > 1 and block == 0:
                    identity = record(
                        f"{prefix}.projection",
                        conv(f"{prefix}.projection", identity, stride, (0, 0, 0, 0)),
                    )
                current = record(f"{prefix}.add", branch + identity)
                current = record(f"{prefix}.relu", functional.relu(current))
        current = record("avgpool", functional.adaptive_avg_pool2d(current, (1, 1)))
        current = record("flatten", torch.flatten(current, 1))
        current = record(
            "logits",
            functional.linear(
                current,
                torch.from_numpy(np.ascontiguousarray(state["fc.weight"])),
                torch.from_numpy(np.ascontiguousarray(state["fc.bias"])),
            ),
        )
    return FloatTrace(maxima=dict(maxima), captures=dict(captures))


def _scale(maximum: float) -> float:
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("calibration maximum must be finite and non-negative")
    return max(maximum / 127.0, 1.0 / (1 << 24))


def _quantization_from_scale(scale: float) -> Quantization:
    if not math.isfinite(scale) or not 0.0 < scale < 1.0:
        raise ValueError(f"activation scale {scale} is outside representable Q1.31")
    multiplier, shift = encode_q31_ratio(scale)
    return Quantization(multiplier, shift, 0)


def _activation_scales(
    maxima: Mapping[str, float],
    quantized_weights: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    scales = {"input": _scale(maxima["input"])}
    scales.update(residual_scale_groups(maxima))
    for stage in range(1, 5):
        for block in range(2):
            prefix = f"layer{stage}.{block}"
            local = _scale(max(maxima[f"{prefix}.conv1"], maxima[f"{prefix}.relu1"]))
            scales[f"{prefix}.conv1"] = local
            scales[f"{prefix}.relu1"] = local
    scales["avgpool"] = scales["layer4.1.relu"]
    scales["flatten"] = scales["layer4.1.relu"]
    scales["logits"] = _scale(maxima["logits"])
    for _iteration in range(4):
        changed = False
        for spec in _conv_specs():
            input_scale = scales[spec.input_id]
            required = input_scale * float(np.max(quantized_weights[spec.command_id][1]))
            if scales[spec.output_id] <= required:
                replacement = float(np.nextafter(required, math.inf))
                scales[spec.output_id] = replacement
                if spec.output_id.endswith(".conv1"):
                    scales[spec.output_id.removesuffix("conv1") + "relu1"] = replacement
                if spec.output_id == "stem.conv":
                    for name in list(scales):
                        if name.startswith(("stem.", "layer1.")) and not name.endswith(
                            ("conv1", "relu1")
                        ):
                            scales[name] = max(scales[name], replacement)
                elif ".conv2" in spec.output_id or ".projection" in spec.output_id:
                    stage = int(spec.output_id[5])
                    for name in list(scales):
                        if name.startswith(f"layer{stage}.") and not name.endswith(
                            ("conv1", "relu1")
                        ):
                            scales[name] = max(scales[name], replacement)
                    if stage == 4:
                        scales["avgpool"] = scales["flatten"] = max(
                            scales["avgpool"], replacement
                        )
                changed = True
        fc_required = scales["flatten"] * float(
            np.max(_quantize_linear_weight(np.asarray(maxima["_fc_weight"]))[1])
        ) if "_fc_weight" in maxima else 0.0
        if fc_required and scales["logits"] <= fc_required:
            scales["logits"] = float(np.nextafter(fc_required, math.inf))
            changed = True
        if not changed:
            break
    for name, scale in scales.items():
        _quantization_from_scale(scale)
    return scales


def _quantize_linear_weight(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = _finite_float_array("fully connected weight", weight, 2)
    maxima = np.max(np.abs(weights), axis=1)
    scales = np.where(maxima > 0.0, maxima / 127.0, 1.0)
    quantized = _round_away_from_zero(weights / scales[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized.T), np.ascontiguousarray(scales)


def _int32_bias(bias: np.ndarray, input_scale: float, weight_scales: np.ndarray) -> np.ndarray:
    values = _round_away_from_zero(
        np.asarray(bias, dtype=np.float64) / (input_scale * weight_scales)
    )
    if np.any(values < INT32_MIN) or np.any(values > INT32_MAX):
        raise ValueError("folded bias exceeds signed INT32 accumulator units")
    return np.ascontiguousarray(values, dtype=np.int32)


def _minimum_bias_scales(bias: np.ndarray, input_scale: float) -> np.ndarray:
    """Return weight-scale floors that keep rounded biases inside INT32."""

    values = np.abs(np.asarray(bias, dtype=np.float64))
    floor = values / (input_scale * (INT32_MAX - 1.0))
    return np.maximum(floor, np.finfo(np.float64).tiny)


def _channel_accumulator_bound(
    weights: np.ndarray, bias: float, input_scale: float, weight_scale: float
) -> tuple[int, np.ndarray, int]:
    quantized_weights = np.clip(
        _round_away_from_zero(np.asarray(weights, dtype=np.float64) / weight_scale),
        -127,
        127,
    ).astype(np.int8)
    quantized_bias_value = int(
        _round_away_from_zero(
            np.array([bias / (input_scale * weight_scale)], dtype=np.float64)
        )[0]
    )
    if not INT32_MIN <= quantized_bias_value <= INT32_MAX:
        return INT32_MAX + 1, quantized_weights, quantized_bias_value
    bound = abs(quantized_bias_value) + 128 * int(
        np.sum(np.abs(quantized_weights.astype(np.int16)), dtype=np.int64)
    )
    return bound, quantized_weights, quantized_bias_value


def _safe_channel_scale(
    weights: np.ndarray, bias: float, input_scale: float, initial_scale: float
) -> float:
    """Find the smallest deterministic scale that passes the export certificate."""

    low = float(initial_scale)
    high = low
    bound, _weights, _bias = _channel_accumulator_bound(
        weights, bias, input_scale, high
    )
    for _attempt in range(64):
        if bound <= INT32_MAX:
            break
        high *= 2.0
        bound, _weights, _bias = _channel_accumulator_bound(
            weights, bias, input_scale, high
        )
    else:
        raise ValueError("cannot certify quantized accumulator channel")
    if high == low:
        return high
    for _attempt in range(64):
        middle = (low + high) * 0.5
        bound, _weights, _bias = _channel_accumulator_bound(
            weights, bias, input_scale, middle
        )
        if bound <= INT32_MAX:
            high = middle
        else:
            low = middle
    return float(np.nextafter(high, math.inf))


def _safe_quantize_conv(
    weight: np.ndarray, bias: np.ndarray, input_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    weights = _finite_float_array("weight", weight, 4)
    biases = _finite_float_array("bias", bias, 1)
    if biases.shape != (weights.shape[0],):
        raise ValueError("bias and convolution output channels differ")
    base = np.max(np.abs(weights), axis=(1, 2, 3)) / 127.0
    base = np.maximum(base, _minimum_bias_scales(biases, input_scale))
    scales = np.empty_like(base)
    quantized = np.empty_like(weights, dtype=np.int8)
    for channel in range(weights.shape[0]):
        scales[channel] = _safe_channel_scale(
            weights[channel], float(biases[channel]), input_scale, float(base[channel])
        )
        bound, channel_weights, _channel_bias = _channel_accumulator_bound(
            weights[channel], float(biases[channel]), input_scale, scales[channel]
        )
        if bound > INT32_MAX:
            raise ValueError(f"cannot certify convolution output channel {channel}")
        quantized[channel] = channel_weights
    return (
        np.ascontiguousarray(np.transpose(quantized, (2, 3, 1, 0))),
        np.ascontiguousarray(scales),
    )


def _safe_quantize_linear(
    weight: np.ndarray, bias: np.ndarray, input_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    weights = _finite_float_array("fully connected weight", weight, 2)
    biases = _finite_float_array("fully connected bias", bias, 1)
    if biases.shape != (weights.shape[0],):
        raise ValueError("bias and fully connected output channels differ")
    base = np.max(np.abs(weights), axis=1) / 127.0
    base = np.maximum(base, _minimum_bias_scales(biases, input_scale))
    scales = np.empty_like(base)
    quantized = np.empty_like(weights, dtype=np.int8)
    for channel in range(weights.shape[0]):
        scales[channel] = _safe_channel_scale(
            weights[channel], float(biases[channel]), input_scale, float(base[channel])
        )
        bound, channel_weights, _channel_bias = _channel_accumulator_bound(
            weights[channel], float(biases[channel]), input_scale, scales[channel]
        )
        if bound > INT32_MAX:
            raise ValueError(f"cannot certify classifier output channel {channel}")
        quantized[channel] = channel_weights
    return np.ascontiguousarray(quantized.T), np.ascontiguousarray(scales)


def build_quantized_resnet18(
    state: Mapping[str, np.ndarray], calibration_inputs: np.ndarray
) -> tuple[QuantizedGraph, np.ndarray, FloatTrace, dict[str, float]]:
    """Build the exact Phase 2A graph for the pinned pretrained architecture."""

    validate_source_arrays(state)
    trace = run_float_resnet18(state, calibration_inputs)
    folded = _folded_convolutions(state)
    quantized_weights = {
        name: quantize_conv_weight(weight)
        for name, (weight, _bias) in folded.items()
    }
    maxima = dict(trace.maxima)
    maxima["_fc_weight"] = np.asarray(state["fc.weight"])
    scales = _activation_scales(maxima, quantized_weights)
    for _iteration in range(2):
        input_ids = {spec.command_id: spec.input_id for spec in _conv_specs()}
        quantized_weights = {
            name: _safe_quantize_conv(weight, bias, scales[input_ids[name]])
            for name, (weight, bias) in folded.items()
        }
        scales = _activation_scales(maxima, quantized_weights)
    tensors: list[TensorSpec] = []
    constants: list[ConstantTensor] = []
    commands = []

    def tensor(name: str, nchw_shape: tuple[int, ...]):
        if len(nchw_shape) == 4:
            shape = (nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1])
            layout = "NHWC"
        else:
            shape = nchw_shape
            layout = "NC"
        tensors.append(TensorSpec(name, shape, layout, _quantization_from_scale(scales[name])))

    tensor("input", (1, 3, 224, 224))
    current_shape = (1, 3, 224, 224)
    spec_by_id = {spec.command_id: spec for spec in _conv_specs()}

    def add_conv(command_id: str, output_shape: tuple[int, ...]):
        nonlocal current_shape
        spec = spec_by_id[command_id]
        q_weight, weight_scales = quantized_weights[command_id]
        folded_bias = folded[command_id][1]
        bias = _int32_bias(folded_bias, scales[spec.input_id], weight_scales)
        weight_name = f"{command_id}.weight"
        bias_name = f"{command_id}.bias"
        constants.append(
            ConstantTensor(
                weight_name,
                tuple(int(value) for value in q_weight.shape),
                "int8",
                "HWIO",
                tuple(int(value) for value in q_weight.reshape(-1)),
            )
        )
        constants.append(
            ConstantTensor(
                bias_name,
                (int(bias.size),),
                "int32",
                "BIAS",
                tuple(int(value) for value in bias),
            )
        )
        tensor(command_id, output_shape)
        ratios = scales[spec.input_id] * weight_scales / scales[spec.output_id]
        encoded = tuple(encode_q31_ratio(float(value)) for value in ratios)
        commands.append(
            Conv2D(
                f"{command_id}.command",
                spec.input_id,
                weight_name,
                spec.output_id,
                tuple(item[0] for item in encoded),
                tuple(item[1] for item in encoded),
                bias_name,
                spec.stride,
                spec.padding,
            )
        )
        current_shape = output_shape

    add_conv("stem.conv", (1, 64, 112, 112))
    tensor("stem.relu", current_shape)
    commands.append(Relu("stem.relu.command", "stem.conv", "stem.relu"))
    tensor("stem.pool", (1, 64, 56, 56))
    commands.append(
        MaxPool(
            "stem.pool.command",
            "stem.relu",
            "stem.pool",
            (3, 3),
            (2, 2),
            (1, 1, 1, 1),
        )
    )
    prior = "stem.pool"
    prior_shape = (1, 64, 56, 56)
    channels = (64, 128, 256, 512)
    for stage, output_channels in enumerate(channels, start=1):
        for block in range(2):
            prefix = f"layer{stage}.{block}"
            stride = 2 if stage > 1 and block == 0 else 1
            spatial = prior_shape[2] // stride
            conv_shape = (1, output_channels, spatial, spatial)
            add_conv(f"{prefix}.conv1", conv_shape)
            tensor(f"{prefix}.relu1", conv_shape)
            commands.append(Relu(f"{prefix}.relu1.command", f"{prefix}.conv1", f"{prefix}.relu1"))
            add_conv(f"{prefix}.conv2", conv_shape)
            residual = prior
            if stage > 1 and block == 0:
                add_conv(f"{prefix}.projection", conv_shape)
                residual = f"{prefix}.projection"
            tensor(f"{prefix}.add", conv_shape)
            commands.append(
                ResidualAdd(
                    f"{prefix}.add.command",
                    f"{prefix}.conv2",
                    residual,
                    f"{prefix}.add",
                )
            )
            tensor(f"{prefix}.relu", conv_shape)
            commands.append(Relu(f"{prefix}.relu.command", f"{prefix}.add", f"{prefix}.relu"))
            prior = f"{prefix}.relu"
            prior_shape = conv_shape
    tensor("avgpool", (1, 512, 1, 1))
    commands.append(GlobalAveragePool("avgpool.command", prior, "avgpool"))
    tensor("flatten", (1, 512))
    commands.append(Flatten("flatten.command", "avgpool", "flatten"))
    fc_weight, fc_scales = _safe_quantize_linear(
        state["fc.weight"], state["fc.bias"], scales["flatten"]
    )
    fc_bias = _int32_bias(state["fc.bias"], scales["flatten"], fc_scales)
    constants.append(
        ConstantTensor(
            "fc.weight",
            fc_weight.shape,
            "int8",
            "IO",
            tuple(int(value) for value in fc_weight.reshape(-1)),
        )
    )
    constants.append(
        ConstantTensor(
            "fc.bias",
            fc_bias.shape,
            "int32",
            "BIAS",
            tuple(int(value) for value in fc_bias),
        )
    )
    tensor("logits", (1, 1000))
    fc_encoded = tuple(
        encode_q31_ratio(float(value))
        for value in scales["flatten"] * fc_scales / scales["logits"]
    )
    commands.append(
        FullyConnected(
            "fc.command",
            "flatten",
            "fc.weight",
            "logits",
            tuple(item[0] for item in fc_encoded),
            tuple(item[1] for item in fc_encoded),
            "fc.bias",
        )
    )
    graph = QuantizedGraph(
        tuple(tensors),
        tuple(constants),
        tuple(commands),
        ("input",),
        ("stem.relu", "layer1.1.relu", "logits"),
    )
    input_values = _round_away_from_zero(calibration_inputs / scales["input"])
    input_values = np.clip(input_values, -127, 127).astype(np.int8)
    input_nhwc = np.ascontiguousarray(np.transpose(input_values, (0, 2, 3, 1)))
    return graph, input_nhwc, trace, scales


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, value: object) -> None:
    data = (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def convert_checkpoint(
    checkpoint_path: str | Path,
    output_prefix: str | Path,
) -> ConversionResult:
    """Convert the pinned checkpoint and publish deterministic NPU files."""

    checkpoint = Path(checkpoint_path).resolve()
    prefix = Path(output_prefix).resolve()
    if not prefix.parent.is_dir():
        raise ValueError("output directory must exist")
    manifest_path = Path(str(prefix) + ".npu.json")
    payload_path = Path(str(prefix) + ".npu.bin")
    input_path = prefix.with_name(f"{prefix.name}.validation.npy")
    provenance_path = prefix.with_name(f"{prefix.name}.conversion.json")
    existing = [
        path
        for path in (manifest_path, payload_path, input_path, provenance_path)
        if path.exists()
    ]
    if existing:
        raise ValueError(f"conversion output already exists: {existing[0].name}")
    state = load_checkpoint(checkpoint)
    calibration = generate_calibration_inputs()
    graph, quantized_inputs, _trace, scales = build_quantized_resnet18(state, calibration)
    package = export_model(graph, prefix)
    input_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=input_path.parent,
            prefix=f".{input_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            input_temporary = Path(stream.name)
            np.save(stream, quantized_inputs[:1], allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(input_temporary, input_path)
        provenance = {
            "architecture": "resnet18",
            "calibration": "deterministic-full-shape-v1",
            "checkpoint": {
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            },
            "evidence_type": REAL_MODEL_HOST_EVIDENCE_TYPE,
            "format": {"major": 1, "minor": 0},
            "input": {
                "bytes": input_path.stat().st_size,
                "sha256": _sha256(input_path),
            },
            "magic": "NPU_RESNET18_CONVERSION",
            "model": {
                "manifest_sha256": _sha256(package.manifest_path),
                "payload_sha256": _sha256(package.payload_path),
            },
            "quantization": {
                "activation": "symmetric-int8-zero-point-0",
                "scale_count": len(scales),
            },
            "source": {"provider": "torchvision", "revision": "IMAGENET1K_V1"},
        }
        _write_canonical(provenance_path, provenance)
    except Exception:
        for path in (
            provenance_path,
            input_path,
            package.manifest_path,
            package.payload_path,
        ):
            if path.exists():
                path.unlink()
        raise
    finally:
        if input_temporary is not None and input_temporary.exists():
            input_temporary.unlink()
    return ConversionResult(package, provenance_path, input_path, graph, scales["input"])
