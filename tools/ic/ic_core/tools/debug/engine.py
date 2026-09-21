"""The query engine behind every debug backend.

`fst.py` and `vcd.py` differ only in how they get a readable VCD on disk. Once
they have one, all four ops are answered here, so the two backends cannot
drift apart in behaviour.

Cycles rather than times are what an agent reasons in, so a clock is resolved
first and rising edges are numbered from zero. Edge times are collected by
streaming the clock identifier alone -- one cheap pass, no full-trace load.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from ...errors import InvalidInput
from . import (
    FirstMismatchOut,
    Point,
    SignalInfo,
    SignalsOut,
    Value,
    ValueAtOut,
    ValueRangeOut,
)
from .vcd_reader import Header, Signal, format_value, iter_changes, parse_header

CLOCK_NAMES = ("clk", "clock", "aclk", "i_clk", "clk_i")


class Wave:
    def __init__(self, path: Path):
        self.path = path
        self.header, self.offset = parse_header(path)
        self._edges: list[int] | None = None
        self._edge_clock: str | None = None

    # -- naming ----------------------------------------------------------

    def resolve(self, name: str) -> Signal:
        signal = self.header.find(name)
        if signal is None:
            raise InvalidInput(
                f"no signal matching {name!r} in {self.path.name}",
                hint="call `signals` to list what the trace contains",
            )
        return signal

    def clock_signal(self, requested: str | None) -> Signal | None:
        if requested:
            return self.resolve(requested)
        candidates = [
            s for s in self.header.signals
            if s.width == 1 and s.leaf.lower() in CLOCK_NAMES
        ]
        if not candidates:
            return None
        # Shallowest wins: the testbench clock, not a clock re-declared inside
        # every instance of the DUT.
        return min(candidates, key=lambda s: (s.path.count("."), len(s.path)))

    # -- cycles ----------------------------------------------------------

    def edges(self, clock: Signal | None, upto: int | None = None) -> list[int]:
        """Times of the clock's rising edges, index == cycle number."""
        if clock is None:
            return []
        if self._edges is not None and self._edge_clock == clock.ident:
            if upto is None or len(self._edges) > upto:
                return self._edges
        times: list[int] = []
        previous = "0"
        for time, _, value in iter_changes(self.path, self.offset, {clock.ident}):
            normalized = value.lstrip("bB").lstrip("0") or "0"
            current = normalized[-1]
            if previous in "0xzXZ" and current == "1":
                times.append(time)
                if upto is not None and len(times) > upto + 1:
                    break
            previous = current
        if upto is None:
            self._edges = times
            self._edge_clock = clock.ident
        return times

    def cycle_time(self, clock: Signal | None, cycle: int) -> int:
        edges = self.edges(clock, upto=cycle)
        if not edges:
            raise InvalidInput(
                "cannot use cycles: no clock found in the trace",
                hint="pass `clock` explicitly, or use `time` instead of `cycle`",
            )
        if cycle >= len(edges):
            raise InvalidInput(
                f"cycle {cycle} is past the end of the trace ({len(edges)} cycles recorded)",
                cycles=len(edges),
            )
        return edges[cycle]

    def time_to_cycle(self, clock: Signal | None, time: int) -> int | None:
        edges = self.edges(clock)
        if not edges:
            return None
        low, high = 0, len(edges) - 1
        if time < edges[0]:
            return None
        while low < high:
            mid = (low + high + 1) // 2
            if edges[mid] <= time:
                low = mid
            else:
                high = mid - 1
        return low

    # -- sampling --------------------------------------------------------

    def values_at(self, signals: list[Signal], time: int) -> dict[str, dict]:
        """Last value of each signal at or before `time`, in one pass."""
        wanted = {s.ident for s in signals}
        latest: dict[str, str] = {}
        for change_time, ident, value in iter_changes(
            self.path, self.offset, wanted, stop_time=time
        ):
            if change_time > time:
                break
            latest[ident] = value
        return {
            s.path: format_value(latest.get(s.ident, "x"), s.width) for s in signals
        }


# -- ops -----------------------------------------------------------------


def op_signals(wave: Wave, params) -> SignalsOut:
    clock = wave.clock_signal(params.clock)
    matches = wave.header.signals
    if params.scope:
        prefix = params.scope.rstrip(".")
        matches = [s for s in matches if s.path == prefix or s.path.startswith(prefix + ".")]
    if params.pattern:
        needle = params.pattern.lower()
        glob = any(ch in needle for ch in "*?[")
        matches = [
            s for s in matches
            if (fnmatch.fnmatch(s.path.lower(), needle) if glob else needle in s.path.lower())
        ]
    total = len(matches)
    shown = matches[: params.limit]
    scopes = wave.header.scopes
    if params.scope:
        prefix = params.scope.rstrip(".")
        scopes = [s for s in scopes if s == prefix or s.startswith(prefix + ".")]
    return SignalsOut(
        scopes=scopes,
        signals=[SignalInfo(path=s.path, width=s.width, kind=s.kind) for s in shown],
        total=total,
        truncated=total > len(shown),
        timescale=wave.header.timescale_text,
        clock=clock.path if clock else None,
    )


def op_value_at(wave: Wave, params) -> ValueAtOut:
    clock = wave.clock_signal(params.clock)
    if params.time is not None:
        time = params.time
    elif params.cycle is not None:
        time = wave.cycle_time(clock, params.cycle)
    else:
        raise InvalidInput("give either `cycle` or `time`")

    resolved: list[Signal] = []
    missing: list[str] = []
    asked: dict[str, str] = {}
    for name in params.signals:
        signal = wave.header.find(name)
        if signal is None:
            missing.append(name)
        else:
            resolved.append(signal)
            asked[signal.path] = name

    by_path = wave.values_at(resolved, time) if resolved else {}
    return ValueAtOut(
        time=time,
        cycle=params.cycle if params.cycle is not None else wave.time_to_cycle(clock, time),
        values={asked[path]: Value(**value) for path, value in by_path.items()},
        missing=missing,
    )


def op_value_range(wave: Wave, params) -> ValueRangeOut:
    clock = wave.clock_signal(params.clock)
    signal = wave.resolve(params.signal)

    start = params.from_time
    if start is None and params.from_cycle is not None:
        start = wave.cycle_time(clock, params.from_cycle)
    end = params.to_time
    if end is None and params.to_cycle is not None:
        end = wave.cycle_time(clock, params.to_cycle)

    points: list[Point] = []
    total = 0
    previous: str | None = None
    for time, _, raw in iter_changes(wave.path, wave.offset, {signal.ident}, stop_time=end):
        if start is not None and time < start:
            previous = raw
            continue
        if end is not None and time > end:
            break
        if params.changes_only and raw == previous:
            continue
        previous = raw
        total += 1
        if len(points) < params.max_points:
            points.append(
                Point(
                    time=time,
                    cycle=wave.time_to_cycle(clock, time),
                    value=Value(**format_value(raw, signal.width)),
                )
            )
    return ValueRangeOut(
        signal=signal.path,
        width=signal.width,
        points=points,
        total=total,
        truncated=total > len(points),
    )


def op_first_mismatch(wave: Wave, params) -> FirstMismatchOut:
    """Walk the clock and stop at the first cycle where ref and dut disagree.

    Sampling happens on rising edges, which is where a synchronous design's
    values are meaningful; comparing at arbitrary times would report the
    combinational glitches between edges as failures.
    """
    clock = wave.clock_signal(params.clock)
    ref = wave.resolve(params.ref)
    dut = wave.resolve(params.dut)
    edges = wave.edges(clock)
    if not edges:
        raise InvalidInput(
            "first_mismatch needs a clock to sample on",
            hint="pass `clock` explicitly if the trace names it unconventionally",
        )

    last_cycle = params.to_cycle if params.to_cycle is not None else len(edges) - 1
    last_cycle = min(last_cycle, len(edges) - 1)
    first_cycle = max(0, params.from_cycle)
    window = {ref.ident, dut.ident}

    # One streaming pass: advance a cursor through the edge times, holding the
    # latest value of each side. Two signals, flat memory, no random access.
    # `history` is capped at the context window, so a ten-million-cycle trace
    # costs the same memory as a ten-cycle one.
    from collections import deque

    history: deque = deque(maxlen=max(1, params.context + 1))
    cursor = first_cycle
    current = {ref.ident: "x", dut.ident: "x"}
    compared = 0
    stop_time = edges[last_cycle]

    def record(cycle: int) -> dict:
        return {
            "cycle": cycle,
            "time": edges[cycle],
            "ref": format_value(current[ref.ident], ref.width),
            "dut": format_value(current[dut.ident], dut.width),
        }

    def disagree(entry: dict) -> bool:
        if params.ignore_unknown and (entry["ref"]["unknown"] or entry["dut"]["unknown"]):
            return False
        return entry["ref"]["bin"] != entry["dut"]["bin"]

    def check(cycle: int):
        entry = record(cycle)
        history.append(entry)
        return entry if disagree(entry) else None

    for time, ident, raw in iter_changes(wave.path, wave.offset, window, stop_time=stop_time):
        # A change at exactly this edge time is applied before the edge is
        # sampled, which is what "value in force at the edge" means.
        while cursor <= last_cycle and edges[cursor] < time:
            compared += 1
            hit = check(cursor)
            if hit:
                return _mismatch(ref, dut, hit, list(history), compared)
            cursor += 1
        current[ident] = raw

    while cursor <= last_cycle:
        compared += 1
        hit = check(cursor)
        if hit:
            return _mismatch(ref, dut, hit, list(history), compared)
        cursor += 1

    return FirstMismatchOut(
        found=False, ref=ref.path, dut=dut.path, compared_cycles=compared
    )


def _mismatch(ref, dut, entry, window, compared) -> FirstMismatchOut:
    return FirstMismatchOut(
        found=True,
        time=entry["time"],
        cycle=entry["cycle"],
        ref=ref.path,
        dut=dut.path,
        ref_value=Value(**entry["ref"]),
        dut_value=Value(**entry["dut"]),
        context=window,
        compared_cycles=compared,
    )


OPS = {
    "signals": op_signals,
    "value_at": op_value_at,
    "value_range": op_value_range,
    "first_mismatch": op_first_mismatch,
}
