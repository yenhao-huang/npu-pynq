"""Bit-accurate numeric contract for the PYNQ NPU.

All arithmetic is integer-only.  Python's unbounded integers are narrowed only
through the explicit validation and saturation operations in this module.
"""

from collections.abc import Sequence

INT8_MIN = -(1 << 7)
INT8_MAX = (1 << 7) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _require_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_range(name: str, value: int, minimum: int, maximum: int) -> int:
    value = _require_integer(name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def saturate_int8(value: int) -> int:
    """Clamp an integer to the signed INT8 interval."""

    value = _require_integer("value", value)
    return max(INT8_MIN, min(INT8_MAX, value))


def saturate_int32(value: int) -> int:
    """Clamp an integer to the signed INT32 interval."""

    value = _require_integer("value", value)
    return max(INT32_MIN, min(INT32_MAX, value))


def mac_int8_int32(accumulator: int, lhs: int, rhs: int) -> int:
    """Multiply signed INT8 operands and saturating-add to an INT32 value."""

    accumulator = _require_range(
        "accumulator", accumulator, INT32_MIN, INT32_MAX
    )
    lhs = _require_range("lhs", lhs, INT8_MIN, INT8_MAX)
    rhs = _require_range("rhs", rhs, INT8_MIN, INT8_MAX)
    return saturate_int32(accumulator + lhs * rhs)


def _round_ratio_away_from_zero(numerator: int, denominator: int) -> int:
    """Round numerator/denominator to nearest, with exact ties away from zero."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    magnitude, remainder = divmod(abs(numerator), denominator)
    if remainder * 2 >= denominator:
        magnitude += 1
    return -magnitude if numerator < 0 else magnitude


def requantize_int32_to_int8(
    accumulator: int,
    multiplier_q31: int,
    shift: int = 0,
    zero_point: int = 0,
) -> int:
    """Apply the ABI Q1.31 requantization rule and return signed INT8."""

    accumulator = _require_range(
        "accumulator", accumulator, INT32_MIN, INT32_MAX
    )
    multiplier_q31 = _require_range(
        "multiplier_q31", multiplier_q31, INT32_MIN, INT32_MAX
    )
    shift = _require_range("shift", shift, 0, 31)
    zero_point = _require_range("zero_point", zero_point, INT8_MIN, INT8_MAX)

    numerator = accumulator * multiplier_q31
    denominator = 1 << (31 + shift)
    rounded = _round_ratio_away_from_zero(numerator, denominator)
    return saturate_int8(rounded + zero_point)


def _validate_int8_matrix(name: str, matrix: Sequence[Sequence[int]]) -> tuple:
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of rows")
    if not matrix:
        raise ValueError(f"{name} must contain at least one row")

    normalized_rows = []
    expected_columns = None
    for row_index, row in enumerate(matrix):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name}[{row_index}] must be a sequence")
        if not row:
            raise ValueError(f"{name}[{row_index}] must not be empty")
        if expected_columns is None:
            expected_columns = len(row)
        elif len(row) != expected_columns:
            raise ValueError(f"{name} must be rectangular")
        normalized_rows.append(
            tuple(
                _require_range(
                    f"{name}[{row_index}][{column_index}]",
                    value,
                    INT8_MIN,
                    INT8_MAX,
                )
                for column_index, value in enumerate(row)
            )
        )
    return tuple(normalized_rows)


def matmul_int8(
    lhs: Sequence[Sequence[int]], rhs: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    """Return dense row-major INT32 output for INT8 matrix multiplication."""

    lhs_rows = _validate_int8_matrix("lhs", lhs)
    rhs_rows = _validate_int8_matrix("rhs", rhs)
    reduction = len(lhs_rows[0])
    if len(rhs_rows) != reduction:
        raise ValueError(
            "lhs column count must equal rhs row count: "
            f"{reduction} != {len(rhs_rows)}"
        )

    columns = len(rhs_rows[0])
    output = []
    for lhs_row in lhs_rows:
        output_row = []
        for column in range(columns):
            accumulator = 0
            for index in range(reduction):
                accumulator = mac_int8_int32(
                    accumulator, lhs_row[index], rhs_rows[index][column]
                )
            output_row.append(accumulator)
        output.append(tuple(output_row))
    return tuple(output)
