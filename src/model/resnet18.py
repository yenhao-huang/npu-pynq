"""Content-addressed assets and structural validation for ResNet-18 acceptance."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
import zipfile

import numpy as np

from .resnet import (
    Conv2D,
    Flatten,
    FullyConnected,
    GlobalAveragePool,
    MaxPool,
    QuantizedGraph,
    Relu,
    ResidualAdd,
)


ACCEPTANCE_MAGIC = "NPU_RESNET18_ACCEPTANCE"
ACCEPTANCE_MAJOR = 1
ACCEPTANCE_MINOR = 0
_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class AcceptanceValidationError(ValueError):
    """Acceptance assets or topology cannot prove the required contract."""


@dataclass(frozen=True)
class AcceptanceAsset:
    filename: str
    byte_count: int
    sha256: str
    path: Path


@dataclass(frozen=True)
class AcceptanceReference:
    framework: str
    version: str
    model_id: str
    preprocessing_id: str


@dataclass(frozen=True)
class AcceptanceThresholds:
    top1_min: float
    exact_output_min: float
    require_cycles: bool


@dataclass(frozen=True)
class AcceptanceDescriptor:
    assets: Mapping[str, AcceptanceAsset]
    reference: AcceptanceReference
    thresholds: AcceptanceThresholds
    class_count: int
    sample_count: int
    capture_tensors: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceCorpus:
    inputs: np.ndarray
    expected_outputs: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    expected_captures: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class AcceptanceBundle:
    descriptor: AcceptanceDescriptor
    corpus: AcceptanceCorpus
    model_manifest_path: Path
    model_payload_path: Path


@dataclass(frozen=True)
class ResNet18Block:
    index: int
    input_tensor: str
    first_convolution: str
    second_convolution: str
    projection_convolution: str | None
    add_command: str
    output_tensor: str


@dataclass(frozen=True)
class ResNet18Topology:
    blocks: tuple[ResNet18Block, ...]
    projection_blocks: tuple[int, ...]
    stem_output: str
    classifier_output: str


def _exact_keys(name: str, value, expected: set[str]) -> dict:
    if not isinstance(value, dict):
        raise AcceptanceValidationError(f"{name} must be an object")
    observed = set(value)
    if observed != expected:
        raise AcceptanceValidationError(
            f"{name} fields mismatch; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AcceptanceValidationError(f"{name} must be a positive integer")
    return value


def _ratio(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AcceptanceValidationError(f"{name} must be finite and in [0, 1]")
    return result


def _nonempty_string(name: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise AcceptanceValidationError(f"{name} must be a non-empty string")
    return value


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise AcceptanceValidationError(f"non-finite JSON value {value!r}")


def _canonical_manifest(path: Path) -> dict:
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceValidationError(
            f"acceptance descriptor cannot be read: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AcceptanceValidationError("acceptance descriptor must be an object")
    canonical = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if data != canonical:
        raise AcceptanceValidationError("acceptance descriptor is not canonical")
    return value


def _asset(root: Path, name: str, value) -> AcceptanceAsset:
    record = _exact_keys(name, value, {"bytes", "filename", "sha256"})
    filename = _nonempty_string(f"{name}.filename", record["filename"])
    if (
        not _FILENAME.fullmatch(filename)
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise AcceptanceValidationError(f"{name}.filename must be a basename")
    byte_count = _positive_integer(f"{name}.bytes", record["bytes"])
    digest = record["sha256"]
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise AcceptanceValidationError(f"{name}.sha256 is malformed")
    path = (root / filename).resolve()
    if path.parent != root.resolve():
        raise AcceptanceValidationError(f"{name}.filename escapes the bundle")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AcceptanceValidationError(f"{name} cannot be read: {error}") from error
    if len(data) != byte_count:
        raise AcceptanceValidationError(f"{name} length mismatch")
    if hashlib.sha256(data).hexdigest() != digest:
        raise AcceptanceValidationError(f"{name} digest mismatch")
    return AcceptanceAsset(filename, byte_count, digest, path)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, order="C", copy=True)
    result.flags.writeable = False
    return result


def _load_corpus(
    path: Path,
    sample_count: int,
    class_count: int,
    capture_names: tuple[str, ...],
) -> AcceptanceCorpus:
    keys = (
        "inputs",
        "expected_outputs",
        "labels",
        "sample_ids",
        *(f"capture_{index}" for index in range(len(capture_names))),
    )
    expected_members = {f"{key}.npy" for key in keys}
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item.filename for item in archive.infolist()]
            if len(members) != len(set(members)):
                raise AcceptanceValidationError("corpus archive has duplicate members")
            if set(members) != expected_members:
                raise AcceptanceValidationError(
                    "corpus members do not match the descriptor captures"
                )
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: _readonly(archive[key]) for key in keys}
    except AcceptanceValidationError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise AcceptanceValidationError(
            f"corpus contains an unsafe object or invalid array: {error}"
        ) from error

    inputs = arrays["inputs"]
    outputs = arrays["expected_outputs"]
    labels = arrays["labels"]
    sample_ids = arrays["sample_ids"]
    captures = {
        name: arrays[f"capture_{index}"]
        for index, name in enumerate(capture_names)
    }
    if inputs.dtype != np.int8 or inputs.ndim != 5 or inputs.shape[1] != 1:
        raise AcceptanceValidationError(
            "corpus inputs must be Sx1xHxWxC signed INT8"
        )
    if outputs.dtype != np.int8 or outputs.ndim < 2:
        raise AcceptanceValidationError(
            "corpus expected_outputs must be signed INT8"
        )
    if labels.dtype != np.int64 or labels.ndim != 1:
        raise AcceptanceValidationError("corpus labels must be rank-one INT64")
    if sample_ids.dtype.kind != "U" or sample_ids.ndim != 1:
        raise AcceptanceValidationError(
            "corpus sample_ids must be a Unicode array without objects"
        )
    all_arrays = (inputs, outputs, labels, sample_ids, *captures.values())
    if any(array.shape[0] != sample_count for array in all_arrays):
        raise AcceptanceValidationError("corpus sample dimensions are inconsistent")
    if outputs.shape[-1] != class_count:
        raise AcceptanceValidationError("expected output class count mismatch")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise AcceptanceValidationError("corpus label is outside the class range")
    identifiers = tuple(str(value) for value in sample_ids)
    if any(not value for value in identifiers) or len(set(identifiers)) != sample_count:
        raise AcceptanceValidationError("corpus sample identifiers are invalid")
    for name, array in captures.items():
        if array.dtype != np.int8 or array.ndim < 2:
            raise AcceptanceValidationError(
                f"capture {name!r} must be signed INT8"
            )
    return AcceptanceCorpus(
        inputs,
        outputs,
        labels,
        sample_ids,
        MappingProxyType(captures),
    )


def load_acceptance_bundle(
    descriptor_path: str | Path,
    *,
    graph: QuantizedGraph | None = None,
) -> AcceptanceBundle:
    """Load and validate all acceptance inputs before runtime construction."""

    descriptor_path = Path(descriptor_path).resolve()
    value = _canonical_manifest(descriptor_path)
    root_record = _exact_keys(
        "descriptor",
        value,
        {
            "assets",
            "capture_tensors",
            "class_count",
            "format",
            "magic",
            "reference",
            "sample_count",
            "thresholds",
        },
    )
    if root_record["magic"] != ACCEPTANCE_MAGIC:
        raise AcceptanceValidationError("acceptance magic is invalid")
    format_record = _exact_keys(
        "format", root_record["format"], {"major", "minor"}
    )
    if (
        format_record["major"] != ACCEPTANCE_MAJOR
        or format_record["minor"] != ACCEPTANCE_MINOR
    ):
        raise AcceptanceValidationError("acceptance format is unsupported")
    class_count = _positive_integer("class_count", root_record["class_count"])
    sample_count = _positive_integer("sample_count", root_record["sample_count"])
    capture_value = root_record["capture_tensors"]
    if not isinstance(capture_value, list) or any(
        not isinstance(name, str) or not name for name in capture_value
    ):
        raise AcceptanceValidationError("capture_tensors must be string names")
    capture_names = tuple(capture_value)
    if len(capture_names) != len(set(capture_names)):
        raise AcceptanceValidationError("capture_tensors contains duplicates")

    reference_record = _exact_keys(
        "reference",
        root_record["reference"],
        {"framework", "model_id", "preprocessing_id", "version"},
    )
    reference = AcceptanceReference(
        **{
            key: _nonempty_string(f"reference.{key}", reference_record[key])
            for key in reference_record
        }
    )
    threshold_record = _exact_keys(
        "thresholds",
        root_record["thresholds"],
        {"exact_output_min", "require_cycles", "top1_min"},
    )
    if not isinstance(threshold_record["require_cycles"], bool):
        raise AcceptanceValidationError("require_cycles must be boolean")
    thresholds = AcceptanceThresholds(
        top1_min=_ratio("top1_min", threshold_record["top1_min"]),
        exact_output_min=_ratio(
            "exact_output_min", threshold_record["exact_output_min"]
        ),
        require_cycles=threshold_record["require_cycles"],
    )
    asset_record = _exact_keys(
        "assets",
        root_record["assets"],
        {"corpus", "model_manifest", "model_payload"},
    )
    root = descriptor_path.parent
    assets = {
        name: _asset(root, f"assets.{name}", asset_record[name])
        for name in sorted(asset_record)
    }
    manifest = assets["model_manifest"].path
    payload = assets["model_payload"].path
    if (
        not manifest.name.endswith(".npu.json")
        or not payload.name.endswith(".npu.bin")
        or manifest.name[:-9] != payload.name[:-8]
    ):
        raise AcceptanceValidationError(
            "model manifest and payload basenames do not match"
        )
    corpus = _load_corpus(
        assets["corpus"].path,
        sample_count,
        class_count,
        capture_names,
    )
    if graph is not None:
        topology = validate_resnet18_topology(graph)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        if len(graph.inputs) != 1 or len(graph.outputs) != 1:
            raise AcceptanceValidationError(
                "acceptance graph must have one input and one output"
            )
        if corpus.inputs.shape[1:] != tensors[graph.inputs[0]].shape:
            raise AcceptanceValidationError("corpus input shape mismatch")
        if corpus.expected_outputs.shape[1:] != tensors[graph.outputs[0]].shape:
            raise AcceptanceValidationError("corpus output shape mismatch")
        if tensors[topology.classifier_output].shape[-1] != class_count:
            raise AcceptanceValidationError("graph class count mismatch")
        produced = {command.output_id for command in graph.commands}
        for name, array in corpus.expected_captures.items():
            if name not in produced or array.shape[1:] != tensors[name].shape:
                raise AcceptanceValidationError(
                    f"capture tensor {name!r} is incompatible with the graph"
                )
    descriptor = AcceptanceDescriptor(
        assets=MappingProxyType(assets),
        reference=reference,
        thresholds=thresholds,
        class_count=class_count,
        sample_count=sample_count,
        capture_tensors=capture_names,
    )
    return AcceptanceBundle(descriptor, corpus, manifest, payload)


def _weight_shape(command: Conv2D, constants) -> tuple[int, ...]:
    return constants[command.weight_id].shape


def validate_resnet18_topology(graph: QuantizedGraph) -> ResNet18Topology:
    """Prove canonical basic-block structure from dependencies and shapes."""

    if not isinstance(graph, QuantizedGraph):
        raise TypeError("graph must be a QuantizedGraph")
    counts = Counter(type(command) for command in graph.commands)
    expected_counts = {
        Conv2D: 20,
        ResidualAdd: 8,
        Relu: 17,
        MaxPool: 1,
        GlobalAveragePool: 1,
        Flatten: 1,
        FullyConnected: 1,
    }
    if counts != expected_counts:
        raise AcceptanceValidationError(
            "graph does not contain the canonical ResNet-18 operator counts"
        )
    commands = graph.commands
    stem_conv, stem_relu, stem_pool = commands[:3]
    if not (
        isinstance(stem_conv, Conv2D)
        and isinstance(stem_relu, Relu)
        and isinstance(stem_pool, MaxPool)
        and stem_relu.input_id == stem_conv.output_id
        and stem_pool.input_id == stem_relu.output_id
    ):
        raise AcceptanceValidationError("canonical stem ordering is invalid")
    constants = {constant.name: constant for constant in graph.constants}
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    if (
        _weight_shape(stem_conv, constants)[:2] != (7, 7)
        or stem_conv.stride != (2, 2)
        or stem_conv.padding != (3, 3, 3, 3)
        or stem_pool.window != (3, 3)
        or stem_pool.stride != (2, 2)
        or stem_pool.padding != (1, 1, 1, 1)
    ):
        raise AcceptanceValidationError("canonical stem parameters are invalid")

    producer = {command.output_id: command for command in commands}
    consumers = defaultdict(list)
    for command in commands:
        references = (
            (command.lhs_id, command.rhs_id)
            if isinstance(command, ResidualAdd)
            else (command.input_id,)
        )
        for reference in references:
            consumers[reference].append(command)

    def main_path(tensor_name):
        second = producer.get(tensor_name)
        if not isinstance(second, Conv2D):
            return None
        activation = producer.get(second.input_id)
        if not isinstance(activation, Relu):
            return None
        first = producer.get(activation.input_id)
        if not isinstance(first, Conv2D) or first.output_id != activation.input_id:
            return None
        return first, activation, second

    blocks = []
    block_input = stem_pool.output_id
    used_convolutions = {stem_conv.command_id}
    used_relus = {stem_relu.command_id}
    projection_blocks = []
    additions = [
        command for command in commands if isinstance(command, ResidualAdd)
    ]
    for index, addition in enumerate(additions):
        candidates = (
            (addition.lhs_id, addition.rhs_id),
            (addition.rhs_id, addition.lhs_id),
        )
        selected = None
        for main_output, shortcut in candidates:
            path = main_path(main_output)
            if path is not None and path[0].input_id == block_input:
                selected = path, shortcut
                break
        if selected is None:
            raise AcceptanceValidationError(
                f"block {index} does not have a two-convolution main branch"
            )
        (first, activation, second), shortcut = selected
        expected_projection = index in (2, 4, 6)
        expected_stride = (2, 2) if expected_projection else (1, 1)
        if (
            _weight_shape(first, constants)[:2] != (3, 3)
            or _weight_shape(second, constants)[:2] != (3, 3)
            or first.padding != (1, 1, 1, 1)
            or second.padding != (1, 1, 1, 1)
            or first.stride != expected_stride
            or second.stride != (1, 1)
        ):
            raise AcceptanceValidationError(
                f"block {index} main convolution parameters are invalid"
            )
        projection_id = None
        if shortcut == block_input:
            if expected_projection:
                raise AcceptanceValidationError(
                    f"block {index} is missing its projection shortcut"
                )
        else:
            projection = producer.get(shortcut)
            if (
                not isinstance(projection, Conv2D)
                or projection.input_id != block_input
                or _weight_shape(projection, constants)[:2] != (1, 1)
            ):
                raise AcceptanceValidationError(
                    f"block {index} projection must use a 1x1 convolution"
                )
            if not expected_projection or projection.stride != expected_stride:
                raise AcceptanceValidationError(
                    f"block {index} has an unexpected projection shortcut"
                )
            projection_id = projection.command_id
            projection_blocks.append(index)
            used_convolutions.add(projection.command_id)
        input_channels = tensors[block_input].shape[-1]
        output_channels = tensors[addition.output_id].shape[-1]
        expected_channels = (
            input_channels * 2 if expected_projection else input_channels
        )
        if output_channels != expected_channels:
            raise AcceptanceValidationError(
                f"block {index} channel transition is invalid"
            )
        post_consumers = consumers[addition.output_id]
        if len(post_consumers) != 1 or not isinstance(post_consumers[0], Relu):
            raise AcceptanceValidationError(
                f"block {index} must end in one residual ReLU"
            )
        post_relu = post_consumers[0]
        blocks.append(
            ResNet18Block(
                index,
                block_input,
                first.command_id,
                second.command_id,
                projection_id,
                addition.command_id,
                post_relu.output_id,
            )
        )
        used_convolutions.update((first.command_id, second.command_id))
        used_relus.update((activation.command_id, post_relu.command_id))
        block_input = post_relu.output_id

    average, flatten, classifier = commands[-3:]
    if not (
        isinstance(average, GlobalAveragePool)
        and isinstance(flatten, Flatten)
        and isinstance(classifier, FullyConnected)
        and average.input_id == block_input
        and flatten.input_id == average.output_id
        and classifier.input_id == flatten.output_id
        and graph.outputs == (classifier.output_id,)
    ):
        raise AcceptanceValidationError("canonical ResNet-18 classifier tail is invalid")
    convolution_ids = {
        command.command_id for command in commands if isinstance(command, Conv2D)
    }
    relu_ids = {
        command.command_id for command in commands if isinstance(command, Relu)
    }
    if used_convolutions != convolution_ids or used_relus != relu_ids:
        raise AcceptanceValidationError("graph contains an unaccounted branch")
    return ResNet18Topology(
        tuple(blocks),
        tuple(projection_blocks),
        stem_pool.output_id,
        classifier.output_id,
    )
