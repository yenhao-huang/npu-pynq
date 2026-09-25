"""A streaming VCD reader.

The constraint that shapes this file: a waveform is hundreds of megabytes and
a debug query needs a few dozen bytes out of it. So nothing here builds a
full-signal database. The header is parsed once and kept (it is small), and
value changes are streamed, filtered to the handful of identifiers the caller
asked about, and discarded.

That is what lets `value_at` on a 184 MB trace answer in one pass with flat
memory instead of loading the trace to look at one signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_SECTION = re.compile(r"\$(\w+)(.*?)\$end", re.DOTALL)

_TIMESCALE_UNITS = {"s": 1e0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15}


@dataclass(frozen=True)
class Signal:
    path: str  #: Dotted hierarchical path, e.g. tb_npu_pe.dut.accumulator
    ident: str  #: The VCD identifier code the value changes are keyed by
    width: int
    kind: str

    @property
    def scope(self) -> str:
        return self.path.rsplit(".", 1)[0] if "." in self.path else ""

    @property
    def leaf(self) -> str:
        return self.path.rsplit(".", 1)[-1]


@dataclass
class Header:
    signals: list[Signal] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    timescale_s: float = 1e-12
    timescale_text: str = "1ps"

    def by_ident(self) -> dict[str, list[Signal]]:
        """One identifier can name several paths: VCD aliases identical nets."""
        out: dict[str, list[Signal]] = {}
        for signal in self.signals:
            out.setdefault(signal.ident, []).append(signal)
        return out

    def find(self, name: str) -> Signal | None:
        """Resolve a signal by full path, or by a unique suffix of one.

        An agent that saw `accumulator` in a `signals` listing should be able
        to pass `accumulator` straight to `value_at` without reconstructing
        `TOP.tb_npu_pe.dut.accumulator` by hand.
        """
        for signal in self.signals:
            if signal.path == name:
                return signal
        suffix_matches = [s for s in self.signals if s.path.endswith("." + name)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        leaf_matches = [s for s in self.signals if s.leaf == name]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        if suffix_matches:
            # Ambiguous: prefer the shallowest, which is the conventional
            # testbench-level view of a signal that also exists inside the DUT.
            return min(suffix_matches, key=lambda s: s.path.count("."))
        return None


def parse_header(path: Path) -> tuple[Header, int]:
    """Read definitions only. Returns the header and the byte offset of `#`."""
    header = Header()
    scope_stack: list[str] = []
    buffer = ""
    offset = 0
    with open(path, "r", errors="replace") as handle:
        while True:
            chunk = handle.read(1 << 16)
            if not chunk:
                break
            buffer += chunk
            end = buffer.find("$enddefinitions")
            search_area = buffer if end == -1 else buffer[: end + 64]
            consumed = 0
            for match in _SECTION.finditer(search_area):
                consumed = match.end()
                keyword, body = match.group(1), match.group(2).strip()
                if keyword == "scope":
                    parts = body.split()
                    if len(parts) >= 2:
                        scope_stack.append(parts[1])
                        header.scopes.append(".".join(scope_stack))
                elif keyword == "upscope":
                    if scope_stack:
                        scope_stack.pop()
                elif keyword == "timescale":
                    header.timescale_text = body.replace(" ", "")
                    header.timescale_s = _timescale(header.timescale_text)
                elif keyword == "var":
                    signal = _var(body, scope_stack)
                    if signal:
                        header.signals.append(signal)
                elif keyword == "enddefinitions":
                    return header, offset + match.end()
            buffer = buffer[consumed:]
            offset += consumed
            if len(buffer) > (1 << 22):  # pathological header; stop growing
                break
    return header, offset


def _timescale(text: str) -> float:
    match = re.match(r"(\d+)\s*([munpf]?s)", text)
    if not match:
        return 1e-12
    return int(match.group(1)) * _TIMESCALE_UNITS.get(match.group(2), 1e-12)


def _var(body: str, scope_stack: list[str]) -> Signal | None:
    parts = body.split()
    if len(parts) < 4:
        return None
    kind, width, ident = parts[0], parts[1], parts[2]
    # The name may be followed by a bit range: `a_in [7:0]`.
    name = parts[3]
    try:
        width_int = int(width)
    except ValueError:
        width_int = 1
    path = ".".join([*scope_stack, name])
    return Signal(path=path, ident=ident, width=width_int, kind=kind)


def iter_changes(path: Path, offset: int, idents: set[str] | None = None,
                 stop_time: int | None = None) -> Iterator[tuple[int, str, str]]:
    """Stream `(time, ident, value)` for the identifiers of interest.

    `idents=None` means every signal, which only `first_mismatch`'s fallback
    and a full export need. Everything else passes a set of two or three.
    """
    time = 0
    with open(path, "r", errors="replace") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            head = line[0]
            if head == "#":
                try:
                    time = int(line[1:])
                except ValueError:
                    continue
                if stop_time is not None and time > stop_time:
                    return
            elif head in "bB":
                value, _, ident = line[1:].partition(" ")
                ident = ident.strip()
                if ident and (idents is None or ident in idents):
                    yield time, ident, value
            elif head in "rR":
                value, _, ident = line[1:].partition(" ")
                ident = ident.strip()
                if ident and (idents is None or ident in idents):
                    yield time, ident, value
            elif head in "01xXzZ":
                ident = line[1:].strip()
                if ident and (idents is None or ident in idents):
                    yield time, ident, line[0]
            elif head == "$":
                continue


def format_value(bits: str, width: int) -> dict:
    """Present one raw VCD value three ways.

    Agents compare decimals and read hex; keeping the raw bits alongside means
    an X or Z is never silently rendered as a number.
    """
    raw = bits.strip()
    if raw and raw[0] in "rR":
        try:
            return {"bin": None, "hex": None, "dec": float(raw[1:]), "signed": float(raw[1:]),
                    "unknown": False, "raw": raw}
        except ValueError:
            pass
    normalized = raw.lstrip("bB")
    unknown = any(c in "xXzZuU-" for c in normalized)
    # VCD left-truncates leading zeros; restore the declared width so hex and
    # sign interpretation line up with the RTL declaration.
    padded = normalized.rjust(width, normalized[0] if unknown and normalized else "0")[-width:] \
        if width else normalized
    if unknown:
        return {"bin": padded, "hex": None, "dec": None, "signed": None,
                "unknown": True, "raw": raw}
    value = int(padded, 2) if padded else 0
    # A 1-bit signal has no meaningful sign: reporting `enable` as -1 because
    # its only bit is set would be technically two's complement and useless.
    signed = (
        value - (1 << width)
        if width > 1 and padded[0] == "1"
        else value
    )
    return {
        "bin": padded,
        "hex": f"0x{value:0{max(1, (width + 3) // 4)}x}",
        "dec": value,
        "signed": signed,
        "unknown": False,
        "raw": raw,
    }
