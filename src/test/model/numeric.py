"""Compatibility re-export of the production numeric contract."""

from src.model.numeric import (
    INT8_MAX,
    INT8_MIN,
    INT32_MAX,
    INT32_MIN,
    mac_int8_int32,
    matmul_int8,
    requantize_int32_to_int8,
    round_ratio_away_from_zero,
    saturate_int8,
    saturate_int32,
    saturating_add_int32,
)

__all__ = [
    "INT8_MAX",
    "INT8_MIN",
    "INT32_MAX",
    "INT32_MIN",
    "mac_int8_int32",
    "matmul_int8",
    "requantize_int32_to_int8",
    "round_ratio_away_from_zero",
    "saturate_int8",
    "saturate_int32",
    "saturating_add_int32",
]
