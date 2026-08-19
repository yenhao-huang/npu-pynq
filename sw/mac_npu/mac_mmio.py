"""PYNQ MMIO driver for the scalar MAC AXI4-Lite peripheral."""

from __future__ import annotations

import time
from pathlib import Path


CONTROL = 0x00
OPERAND_A = 0x04
OPERAND_B = 0x08
STATUS = 0x0C
ACCUMULATOR = 0x10

CONTROL_CLEAR = 1 << 0
CONTROL_START = 1 << 1
STATUS_DONE = 1 << 0


def _encode_int8(value: int) -> int:
    if not isinstance(value, int):
        raise TypeError("MAC operands must be integers")
    if not -128 <= value <= 127:
        raise ValueError(f"MAC operand must fit signed INT8: {value}")
    return value & 0xFF


def _decode_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & (1 << 31) else value


class MacMMIO:
    """Control a mapped MAC peripheral through a PYNQ-compatible MMIO object."""

    def __init__(self, mmio):
        self.mmio = mmio

    def clear(self) -> None:
        self.mmio.write(CONTROL, CONTROL_CLEAR)

    def set_operands(self, a: int, b: int) -> None:
        self.mmio.write(OPERAND_A, _encode_int8(a))
        self.mmio.write(OPERAND_B, _encode_int8(b))

    def start(self) -> None:
        self.mmio.write(CONTROL, CONTROL_START)

    def wait_done(self, timeout: float = 1.0, poll_interval: float = 0.0001) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.mmio.read(STATUS) & STATUS_DONE:
                return
            if poll_interval:
                time.sleep(poll_interval)
        raise TimeoutError("MAC peripheral did not assert done")

    def read_accumulator(self) -> int:
        return _decode_int32(self.mmio.read(ACCUMULATOR))

    def mac(self, a: int, b: int, timeout: float = 1.0) -> int:
        self.set_operands(a, b)
        self.start()
        self.wait_done(timeout=timeout)
        return self.read_accumulator()


def load_mac_overlay(bitfile: str | Path, ip_name: str = "mac_axi_lite_0"):
    """Load an overlay and return ``(overlay, MacMMIO)`` for the named IP."""
    from pynq import MMIO, Overlay

    bitfile = str(Path(bitfile).expanduser().resolve())
    overlay = Overlay(bitfile)
    if ip_name not in overlay.ip_dict:
        available = ", ".join(sorted(overlay.ip_dict))
        raise KeyError(f"IP {ip_name!r} is absent from overlay; available: {available}")

    description = overlay.ip_dict[ip_name]
    base_address = description.get("phys_addr", description.get("base_address"))
    address_range = description.get("addr_range", description.get("range"))
    if base_address is None or address_range is None:
        raise KeyError(f"Overlay metadata for {ip_name!r} lacks address information")

    return overlay, MacMMIO(MMIO(base_address, address_range))
