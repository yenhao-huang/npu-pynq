"""Deterministic Phase 2A exporter for validated quantized ResNet graphs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any

from src.model.numeric import INT32_MAX
from src.model.package import (
    PACKAGE_MAGIC,
    PACKAGE_MAJOR,
    PACKAGE_MINOR,
    REQUIRED_ABI_MAJOR,
    REQUIRED_CAPABILITIES,
    PackageValidationError,
    validate_package_data,
)
from src.model.resnet import (
    ConstantTensor,
    Conv2D,
    Flatten,
    FullyConnected,
    GlobalAveragePool,
    MaxPool,
    QuantizedGraph,
    Relu,
    ResidualAdd,
)

from .planner import MemoryPlanningError, plan_memory


class ExportError(ValueError):
    """A graph cannot be safely or deterministically exported."""


@dataclass(frozen=True)
class AccumulatorCertificate:
    command_id: str
    bounds: tuple[int, ...]


@dataclass(frozen=True)
class ExportedPackage:
    manifest_path: Path
    payload_path: Path
    payload_sha256: str


def _constant_map(graph: QuantizedGraph) -> dict[str, ConstantTensor]:
    return {constant.name: constant for constant in graph.constants}


def _channel_values(constant: ConstantTensor, output_channels: int, channel: int):
    return constant.values[channel::output_channels]


def certify_accumulators(
    graph: QuantizedGraph,
) -> tuple[AccumulatorCertificate, ...]:
    """Prove every matrix output channel is overflow-free for any INT8 input."""

    if not isinstance(graph, QuantizedGraph):
        raise TypeError("graph must be a validated QuantizedGraph")
    constants = _constant_map(graph)
    certificates = []
    for command in graph.commands:
        if not isinstance(command, (Conv2D, FullyConnected)):
            continue
        weight = constants[command.weight_id]
        output_channels = int(weight.shape[-1])
        bias_values = (
            constants[command.bias_id].values
            if command.bias_id is not None
            else (0,) * output_channels
        )
        bounds = []
        for channel in range(output_channels):
            bound = abs(int(bias_values[channel])) + 128 * sum(
                abs(int(value))
                for value in _channel_values(weight, output_channels, channel)
            )
            if bound > INT32_MAX:
                raise ExportError(
                    f"command {command.command_id!r} channel {channel} "
                    f"accumulator bound {bound} exceeds {INT32_MAX}"
                )
            bounds.append(bound)
        certificates.append(
            AccumulatorCertificate(command.command_id, tuple(bounds))
        )
    return tuple(certificates)


def _pack_constants(graph: QuantizedGraph):
    payload = bytearray()
    entries = []
    for constant in sorted(graph.constants, key=lambda item: item.name):
        padding = (-len(payload)) % 64
        payload.extend(b"\x00" * padding)
        offset = len(payload)
        if constant.dtype == "int8":
            encoded = bytes(int(value) & 0xFF for value in constant.values)
        elif constant.dtype == "int32":
            encoded = b"".join(
                struct.pack("<i", int(value)) for value in constant.values
            )
        else:
            raise ExportError(f"unsupported constant dtype {constant.dtype!r}")
        payload.extend(encoded)
        entries.append(
            {
                "dtype": constant.dtype,
                "layout": constant.layout,
                "name": constant.name,
                "offset": offset,
                "shape": list(constant.shape),
                "size": len(encoded),
            }
        )
    return bytes(payload), entries


def _quantization(tensor):
    value = tensor.quantization
    return {
        "multiplier_q31": value.multiplier_q31,
        "shift": value.shift,
        "zero_point": value.zero_point,
    }


def _serialize_command(command) -> dict[str, Any]:
    if isinstance(command, Conv2D):
        return {
            "bias_id": command.bias_id,
            "command_id": command.command_id,
            "dilation": list(command.dilation),
            "groups": command.groups,
            "input_id": command.input_id,
            "multipliers_q31": list(command.multipliers_q31),
            "op": "conv2d",
            "output_id": command.output_id,
            "padding": list(command.padding),
            "shifts": list(command.shifts),
            "stride": list(command.stride),
            "weight_id": command.weight_id,
        }
    if isinstance(command, FullyConnected):
        return {
            "bias_id": command.bias_id,
            "command_id": command.command_id,
            "input_id": command.input_id,
            "multipliers_q31": list(command.multipliers_q31),
            "op": "fully_connected",
            "output_id": command.output_id,
            "shifts": list(command.shifts),
            "weight_id": command.weight_id,
        }
    if isinstance(command, ResidualAdd):
        return {
            "command_id": command.command_id,
            "lhs_id": command.lhs_id,
            "op": "residual_add",
            "output_id": command.output_id,
            "rhs_id": command.rhs_id,
        }
    if isinstance(command, Relu):
        return {
            "command_id": command.command_id,
            "input_id": command.input_id,
            "op": "relu",
            "output_id": command.output_id,
        }
    if isinstance(command, MaxPool):
        return {
            "command_id": command.command_id,
            "input_id": command.input_id,
            "op": "max_pool",
            "output_id": command.output_id,
            "padding": list(command.padding),
            "stride": list(command.stride),
            "window": list(command.window),
        }
    if isinstance(command, GlobalAveragePool):
        return {
            "command_id": command.command_id,
            "input_id": command.input_id,
            "op": "global_average_pool",
            "output_id": command.output_id,
        }
    if isinstance(command, Flatten):
        return {
            "command_id": command.command_id,
            "input_id": command.input_id,
            "op": "flatten",
            "output_id": command.output_id,
        }
    raise ExportError(f"unsupported command {type(command).__name__}")


def _manifest(graph, plan, certificates, constants, payload):
    tensors = [
        {
            "layout": tensor.layout,
            "name": tensor.name,
            "quantization": _quantization(tensor),
            "shape": list(tensor.shape),
        }
        for tensor in sorted(graph.tensors, key=lambda item: item.name)
    ]
    allocations = [
        {
            "allocated_bytes": item.allocated_bytes,
            "first_definition": item.first_definition,
            "last_use": item.last_use,
            "logical_bytes": item.logical_bytes,
            "name": item.name,
            "offset": item.offset,
        }
        for item in plan.allocations
    ]
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "accumulator_certificates": [
            {"bounds": list(item.bounds), "command_id": item.command_id}
            for item in certificates
        ],
        "commands": [_serialize_command(command) for command in graph.commands],
        "constants": constants,
        "format": {"major": PACKAGE_MAJOR, "minor": PACKAGE_MINOR},
        "graph": {
            "inputs": list(graph.inputs),
            "outputs": list(graph.outputs),
        },
        "magic": PACKAGE_MAGIC,
        "memory": {
            "alignment_bytes": 64,
            "allocations": allocations,
            "arena_bytes": plan.arena_bytes,
        },
        "payload_bytes": len(payload),
        "payload_sha256": digest,
        "required_abi": {
            "capabilities": REQUIRED_CAPABILITIES,
            "major": REQUIRED_ABI_MAJOR,
        },
        "tensors": tensors,
    }


def _canonical_json(manifest) -> bytes:
    try:
        encoded = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ExportError(f"manifest serialization failed: {error}") from error
    return (encoded + "\n").encode("utf-8")


def _write_temp(parent: Path, basename: str, data: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=parent,
        prefix=f".{basename}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def _restore(path: Path, old_data: bytes | None) -> None:
    if old_data is None:
        if path.exists():
            path.unlink()
        return
    recovery = _write_temp(path.parent, path.name, old_data)
    try:
        os.replace(recovery, path)
    finally:
        if recovery.exists():
            recovery.unlink()


def export_model(
    graph: QuantizedGraph,
    output_prefix: str | Path,
    *,
    arena_limit_bytes: int | None = None,
) -> ExportedPackage:
    """Validate and atomically publish canonical NAME.npu.json/bin files."""

    if not isinstance(graph, QuantizedGraph):
        raise TypeError("graph must be a validated QuantizedGraph")
    prefix = Path(output_prefix)
    if not prefix.name or not prefix.parent.is_dir():
        raise ExportError("output prefix parent directory must exist")
    manifest_path = Path(str(prefix) + ".npu.json")
    payload_path = Path(str(prefix) + ".npu.bin")
    if manifest_path.exists() != payload_path.exists():
        raise ExportError("existing package pair is incomplete")
    try:
        plan = plan_memory(graph, arena_limit_bytes=arena_limit_bytes)
        certificates = certify_accumulators(graph)
        payload, constants = _pack_constants(graph)
        manifest = _manifest(graph, plan, certificates, constants, payload)
        validate_package_data(manifest, payload)
        manifest_bytes = _canonical_json(manifest)
    except (MemoryPlanningError, PackageValidationError) as error:
        raise ExportError(str(error)) from error

    old_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    old_payload = payload_path.read_bytes() if payload_path.exists() else None
    payload_temp = _write_temp(prefix.parent, payload_path.name, payload)
    manifest_temp = _write_temp(prefix.parent, manifest_path.name, manifest_bytes)
    try:
        os.replace(payload_temp, payload_path)
        os.replace(manifest_temp, manifest_path)
    except OSError as error:
        rollback_errors = []
        for path, data in (
            (payload_path, old_payload),
            (manifest_path, old_manifest),
        ):
            try:
                _restore(path, data)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        detail = (
            f"; rollback errors: {'; '.join(rollback_errors)}"
            if rollback_errors
            else ""
        )
        raise ExportError(f"package publish failed: {error}{detail}") from error
    finally:
        for temporary in (payload_temp, manifest_temp):
            if temporary.exists():
                temporary.unlink()

    return ExportedPackage(
        manifest_path=manifest_path,
        payload_path=payload_path,
        payload_sha256=manifest["payload_sha256"],
    )
