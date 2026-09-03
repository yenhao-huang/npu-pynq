"""Deterministic reduced-shape complete ResNet-18 graph fixtures."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np

from src.model.numeric import INT32_MAX
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


Q = Quantization(INT32_MAX, 0, 0)


def _values(size: int, seed: int) -> tuple[int, ...]:
    return tuple(((index + seed) % 5) - 2 for index in range(size))


def make_reduced_resnet18_graph(prefix: str = "fixture") -> QuantizedGraph:
    tensors = []
    constants = []
    commands = []

    def tensor(name, shape, layout="NHWC"):
        stable = f"{prefix}_{name}"
        tensors.append(TensorSpec(stable, shape, layout, Q))
        return stable

    def weight(name, shape, layout, seed):
        stable = f"{prefix}_{name}"
        constants.append(
            ConstantTensor(
                stable,
                shape,
                "int8",
                layout,
                _values(int(np.prod(shape)), seed),
            )
        )
        return stable

    source = tensor("input", (1, 8, 8, 1))
    stem_weight = weight("stem_weight", (7, 7, 1, 1), "HWIO", 1)
    stem = tensor("stem", (1, 4, 4, 1))
    commands.append(
        Conv2D(
            f"{prefix}_stem_conv",
            source,
            stem_weight,
            stem,
            (INT32_MAX,),
            (0,),
            stride=(2, 2),
            padding=(3, 3, 3, 3),
        )
    )
    stem_relu = tensor("stem_relu", (1, 4, 4, 1))
    commands.append(Relu(f"{prefix}_stem_relu_cmd", stem, stem_relu))
    block_input = tensor("pool", (1, 2, 2, 1))
    commands.append(
        MaxPool(
            f"{prefix}_pool_cmd",
            stem_relu,
            block_input,
            (3, 3),
            (2, 2),
            (1, 1, 1, 1),
        )
    )

    height = width = 2
    input_channels = 1
    seed = 10
    for block_index in range(8):
        projection = block_index in (2, 4, 6)
        stride = (2, 2) if projection else (1, 1)
        output_channels = input_channels * 2 if projection else input_channels
        output_height = (height + 2 - 3) // stride[0] + 1
        output_width = (width + 2 - 3) // stride[1] + 1
        conv1_weight = weight(
            f"block{block_index}_conv1_weight",
            (3, 3, input_channels, output_channels),
            "HWIO",
            seed,
        )
        seed += 1
        conv1 = tensor(
            f"block{block_index}_conv1",
            (1, output_height, output_width, output_channels),
        )
        commands.append(
            Conv2D(
                f"{prefix}_block{block_index}_conv1_cmd",
                block_input,
                conv1_weight,
                conv1,
                (INT32_MAX,) * output_channels,
                (0,) * output_channels,
                stride=stride,
                padding=(1, 1, 1, 1),
            )
        )
        conv1_relu = tensor(
            f"block{block_index}_conv1_relu",
            (1, output_height, output_width, output_channels),
        )
        commands.append(
            Relu(
                f"{prefix}_block{block_index}_conv1_relu_cmd",
                conv1,
                conv1_relu,
            )
        )
        conv2_weight = weight(
            f"block{block_index}_conv2_weight",
            (3, 3, output_channels, output_channels),
            "HWIO",
            seed,
        )
        seed += 1
        conv2 = tensor(
            f"block{block_index}_conv2",
            (1, output_height, output_width, output_channels),
        )
        commands.append(
            Conv2D(
                f"{prefix}_block{block_index}_conv2_cmd",
                conv1_relu,
                conv2_weight,
                conv2,
                (INT32_MAX,) * output_channels,
                (0,) * output_channels,
                padding=(1, 1, 1, 1),
            )
        )
        shortcut = block_input
        if projection:
            projection_weight = weight(
                f"block{block_index}_projection_weight",
                (1, 1, input_channels, output_channels),
                "HWIO",
                seed,
            )
            seed += 1
            shortcut = tensor(
                f"block{block_index}_projection",
                (1, output_height, output_width, output_channels),
            )
            commands.append(
                Conv2D(
                    f"{prefix}_block{block_index}_projection_cmd",
                    block_input,
                    projection_weight,
                    shortcut,
                    (INT32_MAX,) * output_channels,
                    (0,) * output_channels,
                    stride=stride,
                )
            )
        added = tensor(
            f"block{block_index}_added",
            (1, output_height, output_width, output_channels),
        )
        commands.append(
            ResidualAdd(
                f"{prefix}_block{block_index}_add_cmd",
                conv2,
                shortcut,
                added,
            )
        )
        block_output = tensor(
            f"block{block_index}_output",
            (1, output_height, output_width, output_channels),
        )
        commands.append(
            Relu(
                f"{prefix}_block{block_index}_output_relu_cmd",
                added,
                block_output,
            )
        )
        block_input = block_output
        height, width = output_height, output_width
        input_channels = output_channels

    average = tensor("average", (1, 1, 1, input_channels))
    commands.append(
        GlobalAveragePool(f"{prefix}_average_cmd", block_input, average)
    )
    flat = tensor("flat", (1, input_channels), "NC")
    commands.append(Flatten(f"{prefix}_flatten_cmd", average, flat))
    fc_weight = weight(
        "fc_weight", (input_channels, 2), "IO", seed
    )
    logits = tensor("logits", (1, 2), "NC")
    commands.append(
        FullyConnected(
            f"{prefix}_fc_cmd",
            flat,
            fc_weight,
            logits,
            (INT32_MAX, INT32_MAX),
            (0, 0),
        )
    )
    return QuantizedGraph(
        tuple(tensors), tuple(constants), tuple(commands), (source,), (logits,)
    )


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            stream = BytesIO()
            np.lib.format.write_array(stream, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, stream.getvalue())


def write_canonical_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
