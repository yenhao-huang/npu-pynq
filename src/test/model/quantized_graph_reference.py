"""Independent vectorized integer reference for complete quantized graphs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.model.numeric import INT8_MAX, INT8_MIN, INT32_MAX, INT32_MIN
from src.model.resnet import (
    Conv2D,
    Flatten,
    FullyConnected,
    GlobalAveragePool,
    MaxPool,
    QuantizedGraph,
    Relu,
    ResidualAdd,
)


def _requantize(
    accumulators: np.ndarray,
    multipliers_q31: tuple[int, ...],
    shifts: tuple[int, ...],
    zero_point: int,
) -> np.ndarray:
    values = np.asarray(accumulators, dtype=np.int64)
    if np.any(values < INT32_MIN) or np.any(values > INT32_MAX):
        raise ValueError("reference accumulator exceeded signed INT32")
    multipliers = np.asarray(multipliers_q31, dtype=np.int64)
    denominators = np.left_shift(
        np.int64(1), 31 + np.asarray(shifts, dtype=np.int64)
    )
    products = values * multipliers
    magnitudes = (np.abs(products) + denominators // 2) // denominators
    rounded = np.where(products < 0, -magnitudes, magnitudes) + int(zero_point)
    return np.ascontiguousarray(
        np.clip(rounded, INT8_MIN, INT8_MAX), dtype=np.int8
    )


def _conv2d(
    source: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    command: Conv2D,
    output_zero_point: int,
    input_zero_point: int,
) -> np.ndarray:
    top, bottom, left, right = command.padding
    kernel_h, kernel_w, input_channels, output_channels = weights.shape
    padded = np.pad(
        source,
        ((0, 0), (top, bottom), (left, right), (0, 0)),
        constant_values=input_zero_point,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_h, kernel_w), axis=(1, 2)
    )
    windows = windows[
        :, :: command.stride[0], :: command.stride[1], :, :, :
    ]
    patches = np.ascontiguousarray(
        windows.transpose(0, 1, 2, 4, 5, 3).reshape(
            -1, kernel_h * kernel_w * input_channels
        ),
        dtype=np.int32,
    )
    matrix = np.ascontiguousarray(
        weights.reshape(-1, output_channels), dtype=np.int32
    )
    accumulators = patches @ matrix
    accumulators = accumulators.astype(np.int64) + bias.astype(np.int64)
    output = _requantize(
        accumulators,
        command.multipliers_q31,
        command.shifts,
        output_zero_point,
    )
    return output.reshape(
        1, windows.shape[1], windows.shape[2], output_channels
    )


def _fully_connected(
    source: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    command: FullyConnected,
    output_zero_point: int,
) -> np.ndarray:
    accumulators = (
        source.astype(np.int32) @ weights.astype(np.int32)
    ).astype(np.int64) + bias.astype(np.int64)
    return _requantize(
        accumulators,
        command.multipliers_q31,
        command.shifts,
        output_zero_point,
    )


def _max_pool(source: np.ndarray, command: MaxPool) -> np.ndarray:
    top, bottom, left, right = command.padding
    padded = np.pad(
        source,
        ((0, 0), (top, bottom), (left, right), (0, 0)),
        constant_values=INT8_MIN,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, command.window, axis=(1, 2)
    )
    windows = windows[
        :, :: command.stride[0], :: command.stride[1], :, :, :
    ]
    return np.ascontiguousarray(np.max(windows, axis=(-1, -2)), dtype=np.int8)


def _global_average(source: np.ndarray) -> np.ndarray:
    denominator = int(source.shape[1] * source.shape[2])
    totals = np.sum(source, axis=(1, 2), dtype=np.int64)
    magnitudes = (np.abs(totals) + denominator // 2) // denominator
    rounded = np.where(totals < 0, -magnitudes, magnitudes)
    return np.ascontiguousarray(
        np.clip(rounded, INT8_MIN, INT8_MAX)[:, None, None, :],
        dtype=np.int8,
    )


def execute_quantized_graph_reference(
    graph: QuantizedGraph,
    constants: Mapping[str, np.ndarray],
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Execute a validated graph without using runtime lowering code."""

    if set(inputs) != set(graph.inputs):
        raise ValueError("reference inputs do not match graph inputs")
    tensor_specs = {tensor.name: tensor for tensor in graph.tensors}
    values = {
        name: np.ascontiguousarray(value, dtype=np.int8)
        for name, value in inputs.items()
    }
    for command in graph.commands:
        output_spec = tensor_specs[command.output_id]
        if isinstance(command, Conv2D):
            output = _conv2d(
                values[command.input_id],
                constants[command.weight_id],
                constants[command.bias_id],
                command,
                output_spec.quantization.zero_point,
                tensor_specs[command.input_id].quantization.zero_point,
            )
        elif isinstance(command, FullyConnected):
            output = _fully_connected(
                values[command.input_id],
                constants[command.weight_id],
                constants[command.bias_id],
                command,
                output_spec.quantization.zero_point,
            )
        elif isinstance(command, ResidualAdd):
            summed = (
                values[command.lhs_id].astype(np.int16)
                + values[command.rhs_id].astype(np.int16)
                - output_spec.quantization.zero_point
            )
            output = np.ascontiguousarray(
                np.clip(summed, INT8_MIN, INT8_MAX), dtype=np.int8
            )
        elif isinstance(command, Relu):
            output = np.ascontiguousarray(
                np.maximum(
                    values[command.input_id],
                    np.int8(output_spec.quantization.zero_point),
                ),
                dtype=np.int8,
            )
        elif isinstance(command, MaxPool):
            output = _max_pool(values[command.input_id], command)
        elif isinstance(command, GlobalAveragePool):
            output = _global_average(values[command.input_id])
        elif isinstance(command, Flatten):
            output = np.ascontiguousarray(
                values[command.input_id].reshape(output_spec.shape), dtype=np.int8
            )
        else:
            raise TypeError(f"unsupported reference command: {type(command).__name__}")
        if output.shape != output_spec.shape or output.dtype != np.int8:
            raise ValueError(
                f"reference output {command.output_id} violates its tensor spec"
            )
        values[command.output_id] = output
    return {
        name: np.array(values[name], dtype=np.int8, order="C", copy=True)
        for name in graph.outputs
    }
