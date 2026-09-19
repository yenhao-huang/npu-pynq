"""Compatibility re-export of the production performance model."""

from src.model.performance import (
    DEFAULT_TARGET,
    ArrayConfiguration,
    CycleMeasurementAssessment,
    PerformanceAssumptions,
    PerformanceReport,
    ResourceAssessment,
    ResourceEstimate,
    TargetResources,
    assess_cycle_measurement,
    assess_resources,
    estimate_matmul,
)

__all__ = [
    "DEFAULT_TARGET",
    "ArrayConfiguration",
    "CycleMeasurementAssessment",
    "PerformanceAssumptions",
    "PerformanceReport",
    "ResourceAssessment",
    "ResourceEstimate",
    "TargetResources",
    "assess_cycle_measurement",
    "assess_resources",
    "estimate_matmul",
]
