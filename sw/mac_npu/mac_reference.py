"""Dependency-free golden model for the MAC MVP."""


def _require_signed(value: int, width: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must fit signed INT{width}: {value}")


def wrap_signed(value: int, width: int) -> int:
    """Wrap an integer to a signed two's-complement value of ``width`` bits."""
    if width < 1:
        raise ValueError("width must be positive")
    mask = (1 << width) - 1
    wrapped = value & mask
    sign_bit = 1 << (width - 1)
    return wrapped - (1 << width) if wrapped & sign_bit else wrapped


def mac_step(accumulator: int, a: int, b: int, acc_width: int = 32) -> int:
    """Perform one signed INT8 MAC step with wrapping accumulation."""
    _require_signed(a, 8, "a")
    _require_signed(b, 8, "b")
    _require_signed(accumulator, acc_width, "accumulator")
    return wrap_signed(accumulator + a * b, acc_width)


def dot_product(a_values, b_values, acc_width: int = 32) -> int:
    """Accumulate a pair of equally sized signed INT8 iterables."""
    a_values = list(a_values)
    b_values = list(b_values)
    if len(a_values) != len(b_values):
        raise ValueError("operand vectors must have equal lengths")

    accumulator = 0
    for a, b in zip(a_values, b_values):
        accumulator = mac_step(accumulator, a, b, acc_width)
    return accumulator
