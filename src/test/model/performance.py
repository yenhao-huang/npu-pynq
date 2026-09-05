"""Deterministic Phase 0 performance and resource model."""

from dataclasses import dataclass
from math import isfinite


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _positive_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _nonnegative_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


@dataclass(frozen=True)
class TargetResources:
    luts: int
    flip_flops: int
    bram36: int
    dsp48: int

    def __post_init__(self) -> None:
        for name in ("luts", "flip_flops", "bram36", "dsp48"):
            _positive_integer(name, getattr(self, name))


DEFAULT_TARGET = TargetResources(
    luts=53_200,
    flip_flops=106_400,
    bram36=140,
    dsp48=220,
)


@dataclass(frozen=True)
class ArrayConfiguration:
    rows: int
    columns: int
    tile_k: int
    operand_bits: int = 8
    accumulator_bits: int = 32

    def __post_init__(self) -> None:
        for name in ("rows", "columns", "tile_k", "operand_bits", "accumulator_bits"):
            _positive_integer(name, getattr(self, name))


@dataclass(frozen=True)
class PerformanceAssumptions:
    clock_hz: float = 100_000_000.0
    sustained_bandwidth_bytes_per_second: float = 600_000_000.0
    launch_overhead_seconds: float = 20e-6

    def __post_init__(self) -> None:
        _positive_number("clock_hz", self.clock_hz)
        _positive_number(
            "sustained_bandwidth_bytes_per_second",
            self.sustained_bandwidth_bytes_per_second,
        )
        _nonnegative_number("launch_overhead_seconds", self.launch_overhead_seconds)


@dataclass(frozen=True)
class PerformanceReport:
    m: int
    n: int
    k: int
    array: ArrayConfiguration
    assumptions: PerformanceAssumptions
    target: TargetResources
    operations: int
    payload_bytes: int
    tile_count: int
    compute_cycles: int
    compute_seconds: float
    transport_seconds: float
    modeled_seconds: float
    limiting_factor: str
    operations_per_second: float
    payload_bandwidth_bytes_per_second: float
    array_utilization: float


def _tile_extents(size: int, tile: int) -> tuple[int, ...]:
    return tuple(min(tile, size - start) for start in range(0, size, tile))


def estimate_matmul(
    m: int,
    n: int,
    k: int,
    array: ArrayConfiguration,
    assumptions: PerformanceAssumptions = PerformanceAssumptions(),
    target: TargetResources = DEFAULT_TARGET,
) -> PerformanceReport:
    """Estimate a conservatively serialized tiled matrix multiplication."""

    m = _positive_integer("m", m)
    n = _positive_integer("n", n)
    k = _positive_integer("k", k)
    if not isinstance(array, ArrayConfiguration):
        raise TypeError("array must be an ArrayConfiguration")
    if not isinstance(assumptions, PerformanceAssumptions):
        raise TypeError("assumptions must be PerformanceAssumptions")
    if not isinstance(target, TargetResources):
        raise TypeError("target must be TargetResources")

    m_tiles = _tile_extents(m, array.rows)
    n_tiles = _tile_extents(n, array.columns)
    k_tiles = _tile_extents(k, array.tile_k)
    compute_cycles = sum(
        tile_m + tile_n + tile_k - 2
        for tile_m in m_tiles
        for tile_n in n_tiles
        for tile_k in k_tiles
    )
    tile_count = len(m_tiles) * len(n_tiles) * len(k_tiles)
    operations = 2 * m * n * k
    payload_bytes = m * k + k * n + m * n + 9 * n
    compute_seconds = compute_cycles / assumptions.clock_hz
    transport_seconds = (
        payload_bytes / assumptions.sustained_bandwidth_bytes_per_second
    )
    if compute_seconds >= transport_seconds:
        limiting_factor = "compute"
        kernel_seconds = compute_seconds
    else:
        limiting_factor = "bandwidth"
        kernel_seconds = transport_seconds
    modeled_seconds = kernel_seconds + assumptions.launch_overhead_seconds
    peak_operation_slots = 2 * array.rows * array.columns * compute_cycles

    return PerformanceReport(
        m=m,
        n=n,
        k=k,
        array=array,
        assumptions=assumptions,
        target=target,
        operations=operations,
        payload_bytes=payload_bytes,
        tile_count=tile_count,
        compute_cycles=compute_cycles,
        compute_seconds=compute_seconds,
        transport_seconds=transport_seconds,
        modeled_seconds=modeled_seconds,
        limiting_factor=limiting_factor,
        operations_per_second=operations / modeled_seconds,
        payload_bandwidth_bytes_per_second=payload_bytes / modeled_seconds,
        array_utilization=operations / peak_operation_slots,
    )


@dataclass(frozen=True)
class ResourceEstimate:
    luts: int
    flip_flops: int
    bram36: int
    dsp48: int

    def __post_init__(self) -> None:
        for name in ("luts", "flip_flops", "bram36", "dsp48"):
            _nonnegative_integer(name, getattr(self, name))


@dataclass(frozen=True)
class ResourceAssessment:
    estimate: ResourceEstimate
    target: TargetResources
    budget_fraction: float
    utilization_percent: dict[str, float]
    budget_limits: dict[str, float]
    headroom: dict[str, float]
    over_budget: tuple[str, ...]
    passed: bool


def assess_resources(
    estimate: ResourceEstimate,
    target: TargetResources = DEFAULT_TARGET,
    budget_fraction: float = 0.75,
) -> ResourceAssessment:
    if not isinstance(estimate, ResourceEstimate):
        raise TypeError("estimate must be a ResourceEstimate")
    if not isinstance(target, TargetResources):
        raise TypeError("target must be TargetResources")
    fraction = _positive_number("budget_fraction", budget_fraction)
    if fraction > 1.0:
        raise ValueError("budget_fraction must not exceed 1.0")

    names = ("luts", "flip_flops", "bram36", "dsp48")
    utilization = {
        name: getattr(estimate, name) * 100.0 / getattr(target, name)
        for name in names
    }
    limits = {name: getattr(target, name) * fraction for name in names}
    headroom = {name: limits[name] - getattr(estimate, name) for name in names}
    over_budget = tuple(name for name in names if headroom[name] < 0)
    return ResourceAssessment(
        estimate=estimate,
        target=target,
        budget_fraction=fraction,
        utilization_percent=utilization,
        budget_limits=limits,
        headroom=headroom,
        over_budget=over_budget,
        passed=not over_budget,
    )


@dataclass(frozen=True)
class CycleMeasurementAssessment:
    modeled_cycles: int
    measured_cycles: int
    tolerance_percent: float
    delta_cycles: int
    absolute_error_percent: float
    passed: bool


def assess_cycle_measurement(
    modeled_cycles: int,
    measured_cycles: int,
    tolerance_percent: float = 10.0,
) -> CycleMeasurementAssessment:
    modeled_cycles = _positive_integer("modeled_cycles", modeled_cycles)
    measured_cycles = _positive_integer("measured_cycles", measured_cycles)
    tolerance = _nonnegative_number("tolerance_percent", tolerance_percent)
    delta = measured_cycles - modeled_cycles
    absolute_error_percent = abs(delta) * 100.0 / modeled_cycles
    return CycleMeasurementAssessment(
        modeled_cycles=modeled_cycles,
        measured_cycles=measured_cycles,
        tolerance_percent=tolerance,
        delta_cycles=delta,
        absolute_error_percent=absolute_error_percent,
        passed=absolute_error_percent <= tolerance,
    )
