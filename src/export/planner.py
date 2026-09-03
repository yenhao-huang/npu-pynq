"""Deterministic live-interval planning for signed INT8 activations."""

from __future__ import annotations

from dataclasses import dataclass
import math

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


ALIGNMENT_BYTES = 64
MAX_PLAN_BYTES = (1 << 63) - 1


class MemoryPlanningError(ValueError):
    """Tensor lifetimes cannot be represented within the requested arena."""


@dataclass(frozen=True)
class TensorAllocation:
    name: str
    offset: int
    logical_bytes: int
    allocated_bytes: int
    first_definition: int
    last_use: int

    @property
    def end(self) -> int:
        return self.offset + self.allocated_bytes


@dataclass(frozen=True)
class MemoryPlan:
    allocations: tuple[TensorAllocation, ...]
    arena_bytes: int
    alignment_bytes: int = ALIGNMENT_BYTES

    def allocation(self, name: str) -> TensorAllocation:
        for allocation in self.allocations:
            if allocation.name == name:
                return allocation
        raise KeyError(name)


@dataclass(frozen=True)
class _Interval:
    name: str
    logical_bytes: int
    allocated_bytes: int
    first_definition: int
    last_use: int


def _align(value: int) -> int:
    if value < 0 or value > MAX_PLAN_BYTES:
        raise MemoryPlanningError("tensor byte-size overflow")
    remainder = value % ALIGNMENT_BYTES
    aligned = value if remainder == 0 else value + ALIGNMENT_BYTES - remainder
    if aligned > MAX_PLAN_BYTES:
        raise MemoryPlanningError("aligned tensor byte-size overflow")
    return aligned


def _command_inputs(command) -> tuple[str, ...]:
    if isinstance(command, ResidualAdd):
        return command.lhs_id, command.rhs_id
    if isinstance(
        command,
        (Conv2D, FullyConnected, Relu, MaxPool, GlobalAveragePool, Flatten),
    ):
        return (command.input_id,)
    raise MemoryPlanningError(
        f"unsupported command record {type(command).__name__}"
    )


def _intervals(graph: QuantizedGraph) -> tuple[_Interval, ...]:
    definitions = {name: -1 for name in graph.inputs}
    for index, command in enumerate(graph.commands):
        definitions[command.output_id] = index
    last_uses = dict(definitions)
    for index, command in enumerate(graph.commands):
        for name in _command_inputs(command):
            last_uses[name] = max(last_uses[name], index)
    terminal_index = len(graph.commands)
    for name in graph.outputs:
        last_uses[name] = terminal_index

    result = []
    for tensor in graph.tensors:
        logical_bytes = math.prod(tensor.shape)
        if logical_bytes <= 0 or logical_bytes > MAX_PLAN_BYTES:
            raise MemoryPlanningError(
                f"tensor {tensor.name!r} byte-size overflow"
            )
        result.append(
            _Interval(
                name=tensor.name,
                logical_bytes=logical_bytes,
                allocated_bytes=_align(logical_bytes),
                first_definition=definitions[tensor.name],
                last_use=last_uses[tensor.name],
            )
        )
    return tuple(
        sorted(result, key=lambda item: (item.first_definition, item.name))
    )


def _merge_free(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for offset, size in sorted(ranges):
        if not merged:
            merged.append((offset, size))
            continue
        prior_offset, prior_size = merged[-1]
        if prior_offset + prior_size == offset:
            merged[-1] = (prior_offset, prior_size + size)
        else:
            merged.append((offset, size))
    return merged


def plan_memory(
    graph: QuantizedGraph,
    *,
    arena_limit_bytes: int | None = None,
) -> MemoryPlan:
    """Allocate all activation intervals with deterministic aligned first fit."""

    if not isinstance(graph, QuantizedGraph):
        raise TypeError("graph must be a validated QuantizedGraph")
    if arena_limit_bytes is not None:
        if isinstance(arena_limit_bytes, bool) or not isinstance(
            arena_limit_bytes, int
        ):
            raise TypeError("arena_limit_bytes must be an integer")
        if not 1 <= arena_limit_bytes <= MAX_PLAN_BYTES:
            raise ValueError(
                f"arena_limit_bytes must be in [1, {MAX_PLAN_BYTES}]"
            )

    active: list[TensorAllocation] = []
    free_ranges: list[tuple[int, int]] = []
    allocations: list[TensorAllocation] = []
    arena_end = 0

    for interval in _intervals(graph):
        still_active = []
        for allocation in active:
            if allocation.last_use < interval.first_definition:
                free_ranges.append(
                    (allocation.offset, allocation.allocated_bytes)
                )
            else:
                still_active.append(allocation)
        active = still_active
        free_ranges = _merge_free(free_ranges)

        offset = None
        for index, (candidate_offset, candidate_size) in enumerate(free_ranges):
            if candidate_size < interval.allocated_bytes:
                continue
            offset = candidate_offset
            remaining = candidate_size - interval.allocated_bytes
            if remaining:
                free_ranges[index] = (
                    candidate_offset + interval.allocated_bytes,
                    remaining,
                )
            else:
                del free_ranges[index]
            break
        if offset is None:
            offset = _align(arena_end)
            required_end = offset + interval.allocated_bytes
            if required_end > MAX_PLAN_BYTES:
                raise MemoryPlanningError("activation arena size overflow")
            arena_end = required_end

        allocation = TensorAllocation(
            name=interval.name,
            offset=offset,
            logical_bytes=interval.logical_bytes,
            allocated_bytes=interval.allocated_bytes,
            first_definition=interval.first_definition,
            last_use=interval.last_use,
        )
        allocations.append(allocation)
        active.append(allocation)

    if arena_limit_bytes is not None and arena_end > arena_limit_bytes:
        raise MemoryPlanningError(
            f"activation arena requires {arena_end} bytes, "
            f"only {arena_limit_bytes} available"
        )
    return MemoryPlan(
        allocations=tuple(sorted(allocations, key=lambda item: item.name)),
        arena_bytes=arena_end,
    )
