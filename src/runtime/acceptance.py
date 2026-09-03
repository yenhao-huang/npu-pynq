"""Deterministic ResNet-18 acceptance execution and transactional evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from src.model.resnet18 import AcceptanceBundle

from .model import ModelResult, NPUModelRuntime


class AcceptanceRunError(RuntimeError):
    """An acceptance gate failed; success evidence was not published."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise AcceptanceRunError(f"evidence is not canonical JSON: {error}") from error
    return (encoded + "\n").encode("utf-8")


def _publish_atomic(path: Path, data: bytes) -> None:
    if not path.name or not path.parent.is_dir():
        raise AcceptanceRunError("evidence parent directory must exist")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise AcceptanceRunError(f"evidence publication failed: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _clock_value(monotonic_ns: Callable[[], int]) -> int:
    value = monotonic_ns()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AcceptanceRunError("monotonic_ns returned an invalid value")
    return value


def _percentile(values: tuple[int, ...], percentile: int) -> int:
    ordered = sorted(values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, index)]


def _work_signature(result: ModelResult) -> tuple[Any, ...]:
    metrics = result.metrics
    return (
        metrics.command_counts,
        metrics.physical_jobs,
        metrics.mac_count,
        metrics.operation_count,
    )


def _asset_evidence(bundle: AcceptanceBundle) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": asset.byte_count,
            "filename": asset.filename,
            "sha256": asset.sha256,
        }
        for name, asset in sorted(bundle.descriptor.assets.items())
    }


def run_resnet18_acceptance(
    bundle: AcceptanceBundle,
    runtime: NPUModelRuntime,
    *,
    evidence_path: str | Path | None = None,
    mode: str = "host",
    repeat_count: int = 2,
    recovery_probe: Callable[[], None] | None = None,
    provenance: Mapping[str, Any] | None = None,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> Mapping[str, Any]:
    """Run exact acceptance gates and atomically publish evidence on success."""

    if not isinstance(bundle, AcceptanceBundle):
        raise TypeError("bundle must be an AcceptanceBundle")
    if not isinstance(runtime, NPUModelRuntime):
        raise TypeError("runtime must be an NPUModelRuntime")
    if mode not in {"host", "board"}:
        raise ValueError("mode must be 'host' or 'board'")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
        raise TypeError("repeat_count must be an integer")
    if repeat_count < 2:
        raise ValueError("repeat_count must be at least two")
    if recovery_probe is not None and not callable(recovery_probe):
        raise TypeError("recovery_probe must be callable")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    if not callable(monotonic_ns):
        raise TypeError("monotonic_ns must be callable")

    descriptor = bundle.descriptor
    corpus = bundle.corpus
    if len(runtime.input_names) != 1 or len(runtime.output_names) != 1:
        raise AcceptanceRunError("acceptance runtime must have one input and one output")
    input_name = runtime.input_names[0]
    output_name = runtime.output_names[0]
    latencies: list[int] = []
    totals = {
        "physical_jobs": 0,
        "mac_count": 0,
        "operation_count": 0,
        "physical_cycles": 0,
    }
    cycles_available = True
    baseline: dict[str, tuple[np.ndarray, int, tuple[Any, ...]]] = {}
    exact_output_matches = 0
    top1_matches = 0
    invocations = 0
    recovery_injected = False

    run_start = _clock_value(monotonic_ns)
    for pass_index in range(repeat_count):
        if pass_index == 1 and recovery_probe is not None:
            try:
                recovery_probe()
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                recovery_injected = True
            else:
                raise AcceptanceRunError("recovery probe did not inject a failure")

        for sample_index, sample_id_value in enumerate(corpus.sample_ids):
            sample_id = str(sample_id_value)
            start = _clock_value(monotonic_ns)
            result = runtime.run(
                {input_name: corpus.inputs[sample_index]},
                capture_tensors=descriptor.capture_tensors,
            )
            finish = _clock_value(monotonic_ns)
            if finish < start:
                raise AcceptanceRunError("monotonic_ns moved backwards")
            latencies.append(finish - start)
            invocations += 1

            actual = result.outputs[output_name]
            expected = corpus.expected_outputs[sample_index]
            output_exact = bool(np.array_equal(actual, expected))
            prediction = int(np.argmax(actual.reshape(-1)))
            signature = _work_signature(result)
            for tensor_name in descriptor.capture_tensors:
                if not np.array_equal(
                    result.captures[tensor_name],
                    corpus.expected_captures[tensor_name][sample_index],
                ):
                    raise AcceptanceRunError(
                        f"sample {sample_id!r} tensor {tensor_name!r} mismatched"
                    )

            if pass_index == 0:
                exact_output_matches += int(output_exact)
                top1_matches += int(prediction == int(corpus.labels[sample_index]))
                baseline[sample_id] = (actual.copy(), prediction, signature)
            else:
                original_output, original_prediction, original_work = baseline[sample_id]
                if not np.array_equal(actual, original_output):
                    raise AcceptanceRunError(
                        f"sample {sample_id!r} output is not repeatable"
                    )
                if prediction != original_prediction or signature != original_work:
                    raise AcceptanceRunError(
                        f"sample {sample_id!r} prediction or work is not repeatable"
                    )

            metrics = result.metrics
            totals["physical_jobs"] += metrics.physical_jobs
            totals["mac_count"] += metrics.mac_count
            totals["operation_count"] += metrics.operation_count
            if metrics.physical_cycles is None:
                cycles_available = False
            else:
                totals["physical_cycles"] += metrics.physical_cycles

    run_finish = _clock_value(monotonic_ns)
    if run_finish < run_start:
        raise AcceptanceRunError("monotonic_ns moved backwards")
    elapsed_ns = run_finish - run_start
    if elapsed_ns <= 0:
        raise AcceptanceRunError("acceptance duration must be positive")
    exact_ratio = exact_output_matches / descriptor.sample_count
    top1_ratio = top1_matches / descriptor.sample_count
    if exact_ratio < descriptor.thresholds.exact_output_min:
        raise AcceptanceRunError(
            f"exact output ratio {exact_ratio:.6f} is below "
            f"{descriptor.thresholds.exact_output_min:.6f}"
        )
    if top1_ratio < descriptor.thresholds.top1_min:
        raise AcceptanceRunError(
            f"top-1 accuracy {top1_ratio:.6f} is below "
            f"{descriptor.thresholds.top1_min:.6f}"
        )
    if descriptor.thresholds.require_cycles and not cycles_available:
        raise AcceptanceRunError("physical cycle telemetry is required but unavailable")

    logical_input_bytes = corpus.inputs[0].nbytes * invocations
    logical_output_bytes = corpus.expected_outputs[0].nbytes * invocations
    duration_seconds = elapsed_ns / 1_000_000_000
    latency_values = tuple(latencies)
    evidence = {
        "assets": _asset_evidence(bundle),
        "evidence_type": (
            "software-integration" if mode == "host" else "board-acceptance"
        ),
        "format": {"major": 1, "minor": 0},
        "gates": {
            "exact_output_ratio": exact_ratio,
            "repeatability": True,
            "recovery_injected": recovery_injected,
            "top1_accuracy": top1_ratio,
        },
        "magic": "NPU_RESNET18_EVIDENCE",
        "mode": mode,
        "performance": {
            "bandwidth_bytes_per_second": (
                logical_input_bytes + logical_output_bytes
            ) / duration_seconds,
            "elapsed_ns": elapsed_ns,
            "invocations": invocations,
            "latency_ns": {
                "max": max(latency_values),
                "min": min(latency_values),
                "p50": _percentile(latency_values, 50),
                "p95": _percentile(latency_values, 95),
                "p99": _percentile(latency_values, 99),
            },
            "logical_input_bytes": logical_input_bytes,
            "logical_output_bytes": logical_output_bytes,
            "mac_count": totals["mac_count"],
            "operation_count": totals["operation_count"],
            "physical_cycles": (
                totals["physical_cycles"] if cycles_available else None
            ),
            "physical_jobs": totals["physical_jobs"],
            "samples_per_second": invocations / duration_seconds,
        },
        "reference": {
            "framework": descriptor.reference.framework,
            "model_id": descriptor.reference.model_id,
            "preprocessing_id": descriptor.reference.preprocessing_id,
            "version": descriptor.reference.version,
        },
        "sample_count": descriptor.sample_count,
        "sample_ids": [str(value) for value in corpus.sample_ids],
    }
    if provenance is not None:
        try:
            evidence["provenance"] = json.loads(
                _canonical_json(dict(provenance)).decode("utf-8")
            )
        except json.JSONDecodeError as error:
            raise AcceptanceRunError("provenance could not be normalized") from error
    encoded = _canonical_json(evidence)
    if evidence_path is not None:
        _publish_atomic(Path(evidence_path), encoded)
    return MappingProxyType(evidence)
