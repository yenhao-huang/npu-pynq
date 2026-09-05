"""Versioned model-package schema and side-effect-free structural validation."""

from __future__ import annotations

import hashlib


PACKAGE_MAGIC = "NPU_MODEL"
PACKAGE_MAJOR = 1
PACKAGE_MINOR = 0
REQUIRED_ABI_MAJOR = 2
REQUIRED_CAPABILITIES = 0x1F
ALIGNMENT_BYTES = 64
INT32_MAX = (1 << 31) - 1


class PackageValidationError(ValueError):
    """Manifest or payload data violates the Phase 2A package contract."""


def _mapping(name, value):
    if not isinstance(value, dict):
        raise PackageValidationError(f"{name} must be an object")
    return value


def _list(name, value):
    if not isinstance(value, list):
        raise PackageValidationError(f"{name} must be an array")
    return value


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PackageValidationError(
            f"{name} must be an integer no smaller than {minimum}"
        )
    return value


def _unique_names(name, entries):
    names = []
    for index, entry in enumerate(entries):
        entry = _mapping(f"{name}[{index}]", entry)
        value = entry.get("name")
        if not isinstance(value, str) or not value:
            raise PackageValidationError(f"{name}[{index}].name is invalid")
        names.append(value)
    if len(names) != len(set(names)):
        raise PackageValidationError(f"{name} contains duplicate names")
    if names != sorted(names):
        raise PackageValidationError(f"{name} must use stable name order")
    return names


def validate_package_data(manifest: dict, payload: bytes) -> None:
    """Validate an already-parsed manifest and exact packed payload bytes."""

    manifest = _mapping("manifest", manifest)
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if manifest.get("magic") != PACKAGE_MAGIC:
        raise PackageValidationError("package magic is invalid")
    format_record = _mapping("format", manifest.get("format"))
    if format_record.get("major") != PACKAGE_MAJOR:
        raise PackageValidationError("package major version is unsupported")
    _integer("format.minor", format_record.get("minor"))
    required_abi = _mapping("required_abi", manifest.get("required_abi"))
    if required_abi.get("major") != REQUIRED_ABI_MAJOR:
        raise PackageValidationError("required ABI major is unsupported")
    _integer("required_abi.capabilities", required_abi.get("capabilities"))

    declared_bytes = _integer("payload_bytes", manifest.get("payload_bytes"))
    if declared_bytes != len(payload):
        raise PackageValidationError(
            f"payload length {len(payload)} does not match {declared_bytes}"
        )
    digest = manifest.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PackageValidationError("payload digest is malformed")
    observed_digest = hashlib.sha256(payload).hexdigest()
    if digest != observed_digest:
        raise PackageValidationError("payload digest mismatch")

    tensors = _list("tensors", manifest.get("tensors"))
    tensor_names = set(_unique_names("tensors", tensors))
    constants = _list("constants", manifest.get("constants"))
    constant_names = _unique_names("constants", constants)
    occupied = []
    dtype_bytes = {"int8": 1, "int32": 4}
    prior_offset = -1
    for index, entry in enumerate(constants):
        offset = _integer(f"constants[{index}].offset", entry.get("offset"))
        size = _integer(f"constants[{index}].size", entry.get("size"), 1)
        if offset % ALIGNMENT_BYTES:
            raise PackageValidationError("constant offset is misaligned")
        if offset < prior_offset:
            raise PackageValidationError("constant offsets are not stable")
        prior_offset = offset
        shape = entry.get("shape")
        if not isinstance(shape, list) or not shape:
            raise PackageValidationError("constant shape is invalid")
        elements = 1
        for dimension in shape:
            elements *= _integer("constant dimension", dimension, 1)
        element_bytes = dtype_bytes.get(entry.get("dtype"))
        if element_bytes is None or size != elements * element_bytes:
            raise PackageValidationError("constant size does not match dtype/shape")
        end = offset + size
        if end > len(payload):
            raise PackageValidationError("constant range exceeds payload")
        for prior_start, prior_end in occupied:
            if offset < prior_end and prior_start < end:
                raise PackageValidationError("constant ranges overlap")
        occupied.append((offset, end))

    graph = _mapping("graph", manifest.get("graph"))
    for field in ("inputs", "outputs"):
        names = _list(f"graph.{field}", graph.get(field))
        if not names or any(name not in tensor_names for name in names):
            raise PackageValidationError(f"graph {field} reference is invalid")
    commands = _list("commands", manifest.get("commands"))
    command_ids = []
    for index, command in enumerate(commands):
        command = _mapping(f"commands[{index}]", command)
        command_id = command.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise PackageValidationError("command id is invalid")
        command_ids.append(command_id)
        if not isinstance(command.get("op"), str):
            raise PackageValidationError("command op is invalid")
    if len(command_ids) != len(set(command_ids)):
        raise PackageValidationError("commands contain duplicate ids")

    memory = _mapping("memory", manifest.get("memory"))
    arena_bytes = _integer("memory.arena_bytes", memory.get("arena_bytes"), 1)
    allocations = _list("memory.allocations", memory.get("allocations"))
    allocation_names = _unique_names("memory.allocations", allocations)
    if set(allocation_names) != tensor_names:
        raise PackageValidationError("memory allocations do not cover tensors")
    for allocation in allocations:
        offset = _integer("allocation.offset", allocation.get("offset"))
        logical_bytes = _integer(
            "allocation.logical_bytes", allocation.get("logical_bytes"), 1
        )
        allocated_bytes = _integer(
            "allocation.allocated_bytes", allocation.get("allocated_bytes"), 1
        )
        if offset % ALIGNMENT_BYTES or allocated_bytes % ALIGNMENT_BYTES:
            raise PackageValidationError("memory allocation is misaligned")
        if logical_bytes > allocated_bytes or offset + allocated_bytes > arena_bytes:
            raise PackageValidationError("memory allocation exceeds arena")

    certificates = _list(
        "accumulator_certificates", manifest.get("accumulator_certificates")
    )
    certificate_ids = set()
    for index, certificate in enumerate(certificates):
        certificate = _mapping(
            f"accumulator_certificates[{index}]", certificate
        )
        command_id = certificate.get("command_id")
        if command_id not in command_ids or command_id in certificate_ids:
            raise PackageValidationError("accumulator certificate id is invalid")
        certificate_ids.add(command_id)
        bounds = _list("certificate.bounds", certificate.get("bounds"))
        if not bounds:
            raise PackageValidationError("accumulator certificate is empty")
        for bound in bounds:
            value = _integer("accumulator bound", bound)
            if value > INT32_MAX:
                raise PackageValidationError(
                    "accumulator bound exceeds signed INT32"
                )
