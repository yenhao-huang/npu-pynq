"""Comparable INT8 benchmark records for the NPU matrix accelerator.

`acceptance.py` certifies a single run: it gates one execution and publishes a
pass/fail certificate. This module answers the other question, which is whether
two runs may be compared at all and what changed between them.

A record therefore carries the provenance that decides comparability, pairs
every measured matrix point with the analytical model in
`src/model/performance.py`, and states whether its cycle counts were measured on
hardware or modeled on the host. Two records diff into a before/after table only
when their provenance agrees.

Nothing here re-derives latency percentiles or acceptance gates; a model-level
run is summarised from the metrics the runtime already reports.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from src.model.numeric import matmul_int8
from src.model.performance import (
    ArrayConfiguration,
    PerformanceAssumptions,
    estimate_matmul,
)

from .evidence import canonical_json, publish_atomic
from .model import ModelResult

BENCHMARK_MAGIC = "NPU_INT8_BENCHMARK"
BENCHMARK_FORMAT = MappingProxyType({"major": 1, "minor": 0})

MODES = ("simulation", "board")
"""Where the numbers came from. `simulation` never proves board behaviour."""

REQUANTIZATION_SITES = ("host", "hardware")
"""Where INT8 requantization ran. See issue #43; it changes what is measured."""


class BenchmarkError(RuntimeError):
    """A benchmark input was invalid or a measurement cannot be trusted."""


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkError(f"{name} must be a positive integer, got {value!r}")
    return value


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be a non-empty string")
    return value


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise BenchmarkError(f"{name} must be finite")
    return value


def _percentile(values: Sequence[int], percentile: int) -> int:
    ordered = sorted(values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, index)]


@dataclass(frozen=True)
class MatrixPoint:
    """One physical-job shape to measure."""

    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            _positive_integer(name, getattr(self, name))

    @property
    def label(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


@dataclass(frozen=True)
class Provenance:
    """What must match before two records may be compared.

    `commit` and `branch` identify the source. `mode` separates host modelling
    from board measurement. `array` and `clock_hz` fix the hardware shape.
    `requantization_site` records whether INT8 requantization ran on the host or
    in hardware, because a host-requantized run measures different work.
    """

    commit: str
    branch: str
    mode: str
    array: ArrayConfiguration
    requantization_site: str
    clock_hz: float
    tool_versions: Mapping[str, str] = MappingProxyType({})
    notes: str = ""

    def __post_init__(self) -> None:
        _text("commit", self.commit)
        _text("branch", self.branch)
        if self.mode not in MODES:
            raise BenchmarkError(f"mode must be one of {MODES}")
        if not isinstance(self.array, ArrayConfiguration):
            raise BenchmarkError("array must be an ArrayConfiguration")
        if self.requantization_site not in REQUANTIZATION_SITES:
            raise BenchmarkError(
                f"requantization_site must be one of {REQUANTIZATION_SITES}"
            )
        clock = _finite("clock_hz", self.clock_hz)
        if clock <= 0.0:
            raise BenchmarkError("clock_hz must be positive")
        if not isinstance(self.tool_versions, Mapping):
            raise BenchmarkError("tool_versions must be a mapping")
        for key, value in self.tool_versions.items():
            _text("tool_versions key", key)
            _text(f"tool_versions[{key}]", value)
        if not isinstance(self.notes, str):
            raise BenchmarkError("notes must be a string")

    def as_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "array": {
                    "accumulator_bits": self.array.accumulator_bits,
                    "columns": self.array.columns,
                    "operand_bits": self.array.operand_bits,
                    "rows": self.array.rows,
                    "tile_k": self.array.tile_k,
                },
                "branch": self.branch,
                "clock_hz": self.clock_hz,
                "commit": self.commit,
                "mode": self.mode,
                "notes": self.notes,
                "requantization_site": self.requantization_site,
                "tool_versions": dict(sorted(self.tool_versions.items())),
            }
        )


COMPARABILITY_KEYS = (
    "array",
    "mode",
    "requantization_site",
    "clock_hz",
)
"""Provenance fields that must agree before a before/after diff means anything."""


class HostMatrixEngine:
    """Exact INT8 matrix engine whose cycle counts are modeled, not measured.

    Stands in for `NPURuntime` so a sweep runs without a board. Results are
    bit-exact against `src/model/numeric.py`; the reported cycle count is the
    serialized minimum documented in `docs/manual/hw/matrx_controller.md`
    section 7:

        M*K + K*N + M*N + K + M + N + 1

    That figure assumes no AXI-Stream stall and no overlap between stages, so it
    is a floor rather than a prediction. Records built from this engine are
    marked `cycles_measured: false`.
    """

    cycles_measured = False

    def __init__(self, *, max_m: int, max_n: int, max_k: int) -> None:
        self.max_m = _positive_integer("max_m", max_m)
        self.max_n = _positive_integer("max_n", max_n)
        self.max_k = _positive_integer("max_k", max_k)
        self.last_metrics: Any = None

    @staticmethod
    def modeled_cycles(m: int, n: int, k: int) -> int:
        return m * k + k * n + m * n + k + m + n + 1

    def run(self, a_matrix: np.ndarray, b_matrix: np.ndarray, **_timeouts: Any):
        a = np.asarray(a_matrix)
        b = np.asarray(b_matrix)
        if a.dtype != np.int8 or b.dtype != np.int8:
            raise BenchmarkError("host engine operands must be int8")
        if a.ndim != 2 or b.ndim != 2:
            raise BenchmarkError("host engine operands must be two-dimensional")
        m, k = a.shape
        k_b, n = b.shape
        if k != k_b:
            raise BenchmarkError("operand reduction dimensions disagree")
        if m > self.max_m or n > self.max_n or k > self.max_k:
            raise BenchmarkError("job exceeds the configured physical limits")
        product = matmul_int8(a.tolist(), b.tolist())
        self.last_metrics = _HostMetrics(self.modeled_cycles(m, n, k))
        return np.asarray(product, dtype=np.int32)


@dataclass(frozen=True)
class _HostMetrics:
    cycles: int


def _deterministic_operands(point: MatrixPoint, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    a = generator.integers(-128, 128, size=(point.m, point.k), dtype=np.int64)
    b = generator.integers(-128, 128, size=(point.k, point.n), dtype=np.int64)
    return a.astype(np.int8), b.astype(np.int8)


def measure_matrix_point(
    engine: Any,
    point: MatrixPoint,
    *,
    array: ArrayConfiguration,
    assumptions: PerformanceAssumptions = PerformanceAssumptions(),
    repeat_count: int = 3,
    seed: int = 0,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> Mapping[str, Any]:
    """Measure one matrix shape and pair it with the analytical model.

    Every repetition is checked against the golden product and against the first
    repetition, so an unrepeatable or wrong result fails instead of being
    reported as a fast one.
    """

    if not isinstance(point, MatrixPoint):
        raise BenchmarkError("point must be a MatrixPoint")
    if not isinstance(array, ArrayConfiguration):
        raise BenchmarkError("array must be an ArrayConfiguration")
    if not isinstance(assumptions, PerformanceAssumptions):
        raise BenchmarkError("assumptions must be PerformanceAssumptions")
    _positive_integer("repeat_count", repeat_count)
    if repeat_count < 2:
        raise BenchmarkError("repeat_count must be at least two")
    if not callable(monotonic_ns):
        raise BenchmarkError("monotonic_ns must be callable")

    a_matrix, b_matrix = _deterministic_operands(point, seed)
    golden = np.asarray(matmul_int8(a_matrix.tolist(), b_matrix.tolist()), dtype=np.int32)

    latencies: list[int] = []
    cycles: list[int] = []
    cycles_available = True
    first: np.ndarray | None = None

    for _ in range(repeat_count):
        start = monotonic_ns()
        observed = engine.run(a_matrix, b_matrix)
        finish = monotonic_ns()
        if not isinstance(start, int) or not isinstance(finish, int) or finish < start:
            raise BenchmarkError("monotonic_ns did not advance monotonically")
        latencies.append(finish - start)

        observed = np.asarray(observed)
        if observed.dtype != np.int32:
            raise BenchmarkError(
                f"{point.label}: engine returned {observed.dtype}, expected int32"
            )
        if not np.array_equal(observed, golden):
            raise BenchmarkError(f"{point.label}: result disagrees with the golden model")
        if first is None:
            first = observed.copy()
        elif not np.array_equal(observed, first):
            raise BenchmarkError(f"{point.label}: result is not repeatable")

        metrics = getattr(engine, "last_metrics", None)
        value = getattr(metrics, "cycles", None)
        if value is None:
            cycles_available = False
        else:
            cycles.append(int(value))

    model = estimate_matmul(point.m, point.n, point.k, array, assumptions)
    measured_cycles = min(cycles) if cycles_available and cycles else None
    record: dict[str, Any] = {
        "cycles": {
            "measured": measured_cycles,
            "modeled_compute": model.compute_cycles,
            "ratio": (
                measured_cycles / model.compute_cycles
                if measured_cycles is not None and model.compute_cycles
                else None
            ),
        },
        "k": point.k,
        "latency_ns": {
            "max": max(latencies),
            "min": min(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "m": point.m,
        "model": {
            "array_utilization": model.array_utilization,
            "limiting_factor": model.limiting_factor,
            "modeled_seconds": model.modeled_seconds,
            "operations": model.operations,
            "payload_bytes": model.payload_bytes,
            "tile_count": model.tile_count,
        },
        "n": point.n,
        "repeat_count": repeat_count,
        "seed": seed,
        "shape": point.label,
    }
    return MappingProxyType(record)


def run_matrix_sweep(
    engine: Any,
    points: Sequence[MatrixPoint],
    *,
    array: ArrayConfiguration,
    assumptions: PerformanceAssumptions = PerformanceAssumptions(),
    repeat_count: int = 3,
    seed: int = 0,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[Mapping[str, Any], ...]:
    """Measure every point in order, failing on the first untrustworthy result."""

    if not points:
        raise BenchmarkError("sweep requires at least one point")
    labels = [point.label for point in points]
    if len(set(labels)) != len(labels):
        raise BenchmarkError("sweep points must be distinct")
    return tuple(
        measure_matrix_point(
            engine,
            point,
            array=array,
            assumptions=assumptions,
            repeat_count=repeat_count,
            seed=seed + index,
            monotonic_ns=monotonic_ns,
        )
        for index, point in enumerate(points)
    )


def _float_matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise BenchmarkError(f"{name} must be two-dimensional (samples, classes)")
    if array.size == 0:
        raise BenchmarkError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise BenchmarkError(f"{name} must be finite")
    return array


def _top_k_indices(row: np.ndarray, k: int) -> set[int]:
    order = np.argsort(-row, kind="stable")
    return {int(index) for index in order[:k]}


def compare_int8_to_float(
    quantized_outputs: Any,
    float_outputs: Any,
    *,
    scale: float,
    zero_point: int = 0,
    top_k: Sequence[int] = (1, 5),
) -> Mapping[str, Any]:
    """Compare a dequantized INT8 result against its own float baseline.

    This is the delta that acceptance does not measure: acceptance compares
    against a fixed expected corpus and label top-1, which cannot say how much
    the INT8 path moved relative to the float model it came from.

    `quantized_outputs` is shaped (samples, classes) and holds INT8 logits;
    `float_outputs` holds the float baseline for the same samples.
    """

    quantized = np.asarray(quantized_outputs)
    if quantized.dtype != np.int8:
        raise BenchmarkError("quantized_outputs must be int8")
    quantized = _float_matrix("quantized_outputs", quantized)
    baseline = _float_matrix("float_outputs", float_outputs)
    if quantized.shape != baseline.shape:
        raise BenchmarkError("quantized and float outputs must have the same shape")
    scale_value = _finite("scale", scale)
    if scale_value <= 0.0:
        raise BenchmarkError("scale must be positive")
    if isinstance(zero_point, bool) or not isinstance(zero_point, int):
        raise BenchmarkError("zero_point must be an integer")
    levels = tuple(sorted({_positive_integer("top_k entry", value) for value in top_k}))
    if not levels:
        raise BenchmarkError("top_k must contain at least one level")
    classes = quantized.shape[1]
    if max(levels) > classes:
        raise BenchmarkError("top_k exceeds the number of classes")

    dequantized = (quantized - float(zero_point)) * scale_value
    difference = np.abs(dequantized - baseline)

    agreement: dict[str, float] = {}
    for level in levels:
        matches = sum(
            1
            for index in range(quantized.shape[0])
            if _top_k_indices(dequantized[index], level)
            == _top_k_indices(baseline[index], level)
        )
        agreement[f"top{level}"] = matches / quantized.shape[0]

    top1_predictions_match = sum(
        1
        for index in range(quantized.shape[0])
        if int(np.argmax(dequantized[index])) == int(np.argmax(baseline[index]))
    )

    return MappingProxyType(
        {
            "classes": classes,
            "dequantization": {"scale": scale_value, "zero_point": zero_point},
            "logit_difference": {
                "max_absolute": float(difference.max()),
                "mean_absolute": float(difference.mean()),
            },
            "sample_count": int(quantized.shape[0]),
            "set_agreement": dict(sorted(agreement.items())),
            "top1_prediction_agreement": top1_predictions_match / quantized.shape[0],
        }
    )


def summarize_model_run(result: ModelResult) -> Mapping[str, Any]:
    """Summarise one model execution from the metrics the runtime already reports."""

    if not isinstance(result, ModelResult):
        raise BenchmarkError("result must be a ModelResult")
    metrics = result.metrics
    return MappingProxyType(
        {
            "command_counts": dict(sorted(dict(metrics.command_counts).items())),
            "elapsed_seconds": float(metrics.elapsed_seconds),
            "mac_count": int(metrics.mac_count),
            "operation_count": int(metrics.operation_count),
            "physical_cycles": (
                None if metrics.physical_cycles is None else int(metrics.physical_cycles)
            ),
            "physical_jobs": int(metrics.physical_jobs),
        }
    )


def build_record(
    *,
    provenance: Provenance,
    matrix_sweep: Sequence[Mapping[str, Any]] = (),
    model_runs: Mapping[str, Mapping[str, Any]] | None = None,
    accuracy: Mapping[str, Any] | None = None,
    cycles_measured: bool,
) -> Mapping[str, Any]:
    """Assemble one benchmark record.

    `cycles_measured` states plainly whether cycle counts came from hardware.
    A record built from `HostMatrixEngine` must pass `False`; reporting modeled
    cycles as measured would make every later comparison meaningless.
    """

    if not isinstance(provenance, Provenance):
        raise BenchmarkError("provenance must be a Provenance")
    if not isinstance(cycles_measured, bool):
        raise BenchmarkError("cycles_measured must be a boolean")
    if cycles_measured and provenance.mode != "board":
        raise BenchmarkError("only a board run may claim measured cycles")
    if model_runs is not None and not isinstance(model_runs, Mapping):
        raise BenchmarkError("model_runs must be a mapping")
    if accuracy is not None and not isinstance(accuracy, Mapping):
        raise BenchmarkError("accuracy must be a mapping")
    if not matrix_sweep and not model_runs:
        raise BenchmarkError("a record must contain a matrix sweep or a model run")

    record = {
        "accuracy": None if accuracy is None else dict(accuracy),
        "cycles_measured": cycles_measured,
        "format": dict(BENCHMARK_FORMAT),
        "magic": BENCHMARK_MAGIC,
        "matrix_sweep": [dict(point) for point in matrix_sweep],
        "model_runs": (
            {} if model_runs is None else {key: dict(value) for key, value in model_runs.items()}
        ),
        "provenance": dict(provenance.as_mapping()),
    }
    canonical_json(record, BenchmarkError)
    return MappingProxyType(record)


def publish_record(path: str | Path, record: Mapping[str, Any]) -> bytes:
    """Write ``record`` atomically as canonical JSON and return the bytes written."""

    encoded = canonical_json(dict(record), BenchmarkError)
    publish_atomic(Path(path), encoded, BenchmarkError)
    return encoded


def load_record(path: str | Path) -> Mapping[str, Any]:
    """Read a published record, rejecting anything that is not one."""

    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise BenchmarkError(f"benchmark record could not be read: {error}") from error
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"benchmark record is not JSON: {error}") from error
    if not isinstance(record, dict) or record.get("magic") != BENCHMARK_MAGIC:
        raise BenchmarkError("file is not an NPU benchmark record")
    return MappingProxyType(record)


def _comparability(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[str, ...]:
    before_provenance = before.get("provenance", {})
    after_provenance = after.get("provenance", {})
    differences = [
        key
        for key in COMPARABILITY_KEYS
        if before_provenance.get(key) != after_provenance.get(key)
    ]
    if before.get("cycles_measured") != after.get("cycles_measured"):
        differences.append("cycles_measured")
    return tuple(sorted(differences))


def compare_records(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Diff two records into a before/after table.

    Comparability is reported, never assumed. When the two records disagree on
    array shape, mode, clock, requantization site, or whether cycles were
    measured, `comparable` is false and the blocking fields are named; the
    per-shape deltas are still produced so the difference can be inspected, but
    they must not be quoted as a speedup.
    """

    for name, record in (("before", before), ("after", after)):
        if not isinstance(record, Mapping) or record.get("magic") != BENCHMARK_MAGIC:
            raise BenchmarkError(f"{name} is not an NPU benchmark record")

    blocking = _comparability(before, after)
    before_points = {point["shape"]: point for point in before.get("matrix_sweep", [])}
    after_points = {point["shape"]: point for point in after.get("matrix_sweep", [])}
    shared = sorted(set(before_points) & set(after_points))

    shapes = []
    for shape in shared:
        old = before_points[shape]
        new = after_points[shape]
        old_cycles = old["cycles"]["measured"]
        new_cycles = new["cycles"]["measured"]
        shapes.append(
            {
                "cycles": {
                    "after": new_cycles,
                    "before": old_cycles,
                    "speedup": (
                        old_cycles / new_cycles
                        if old_cycles is not None and new_cycles
                        else None
                    ),
                },
                "latency_p50_ns": {
                    "after": new["latency_ns"]["p50"],
                    "before": old["latency_ns"]["p50"],
                },
                "shape": shape,
            }
        )

    return MappingProxyType(
        {
            "blocking_differences": list(blocking),
            "comparable": not blocking,
            "only_in_after": sorted(set(after_points) - set(before_points)),
            "only_in_before": sorted(set(before_points) - set(after_points)),
            "shapes": shapes,
        }
    )
