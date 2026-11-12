"""Profile one run: engine stages and delegate phases.

An engine wraps each of its stages (tokenize, prefill, decode, ...) in
``profiler.stage(name)``; stages that call the package pass ``model=True``.
Attached to an ExportedPackage, the profiler also turns on the delegate
backends' phase counters (for the NPU: pack_a, stage_b, pl, unpack_c per
program) and collects them when it detaches. The report then splits the time
spent in package calls into delegate time and the rest (model.so's CPU code
and dispatch).

Nothing here knows a task or a device: phase names, byte counts and the
device clock come from the backends.

Outputs:
  profile.json  run metadata, stage and phase aggregates, summary
  trace.json    with ``events`` > 0: Chrome trace events, the JSON format
                PyTorch's Kineto writes; open in https://ui.perfetto.dev

Times use time.perf_counter_ns, which is CLOCK_MONOTONIC on Linux, the clock
libnpu_rt stamps its events with, so both share one timeline.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator

FORMAT = 'pynqo-profile/1'
PHASES = ('pack_a', 'stage_b', 'pl', 'unpack_c')


@dataclass
class Aggregate:
    count: int = 0
    total_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0

    def add(self, ns: int) -> None:
        self.min_ns = min(self.min_ns, ns) if self.count else ns
        self.max_ns = max(self.max_ns, ns)
        self.count += 1
        self.total_ns += ns

    def as_dict(self) -> dict[str, float]:
        mean = self.total_ns / self.count if self.count else 0.0
        return {'count': self.count, 'total_ns': self.total_ns, 'mean_ns': round(mean, 1),
                'min_ns': self.min_ns, 'max_ns': self.max_ns}


class Profiler:
    """Stage and phase aggregates for one run; see the module docstring."""

    def __init__(self, clock: Callable[[], int] = time.perf_counter_ns, events: int = 0,
                 meta: dict[str, Any] | None = None) -> None:
        self.clock = clock
        self.events = events          # timeline capacity per source; 0 keeps aggregates only
        self.meta = dict(meta or {})
        self.stages: dict[str, Aggregate] = {}
        self.model_stages: set[str] = set()
        self.stage_events: list[tuple[str, int, int, dict]] = []
        self.delegates: dict[str, dict[str, Any]] = {}
        self.package = None
        self.start_ns = clock()

    @contextmanager
    def stage(self, name: str, model: bool = False, **args: Any) -> Iterator[None]:
        begin = self.clock()
        try:
            yield
        finally:
            end = self.clock()
            self.stages.setdefault(name, Aggregate()).add(end - begin)
            if model:
                self.model_stages.add(name)
            if self.events and len(self.stage_events) < self.events:
                self.stage_events.append((name, begin, end, args))

    def attach(self, package: Any) -> None:
        """Clear and start the package's delegate phase counters."""
        self.package = package
        package.profile(True, self.events)

    def detach(self) -> None:
        """Collect the delegate phases and stop the counters."""
        if self.package is None:
            return
        stats, events = self.package.phase_stats(), self.package.events()
        clocks = {name: getattr(b, 'clock_hz', None) for name, b in self.package.backends.items()}
        for device, programs in stats.items():
            recorded, dropped = events.get(device, ([], 0))
            self.delegates[device] = {'clock_hz': clocks.get(device), 'programs': programs,
                                      'events': recorded, 'events_dropped': dropped}
        self.package.profile(False)
        self.package = None

    # -- report ---------------------------------------------------------------

    def delegate_report(self, device: str) -> dict[str, Any]:
        data = self.delegates[device]
        clock_hz = data['clock_hz']
        programs, totals = [], {'calls': 0, 'call_ns': 0, 'pl_cycles': 0, 'polls': 0, 'jobs': 0}
        for p in data['programs']:
            if not p['calls']:
                continue
            row = dict(p)
            row['call_ns'] = sum(p[f'{phase}_ns'] for phase in PHASES)
            for phase in PHASES:   # bytes per ns is GB/s; pl's bytes need PL counters
                nbytes, ns = p.get(f'{phase}_bytes'), p[f'{phase}_ns']
                row[f'{phase}_gbps'] = round(nbytes / ns, 4) if nbytes is not None and ns else None
            row['macs'] = p['m'] * p['n'] * p['k'] * p['calls']
            row['pl_busy_ns'] = round(p['pl_cycles'] * 1e9 / clock_hz) if clock_hz and p['pl_cycles'] else None
            programs.append(row)
            for key in ('calls', 'call_ns', 'pl_cycles', 'polls', 'jobs'):
                totals[key] += row[key]
            for phase in PHASES:
                totals[f'{phase}_ns'] = totals.get(f'{phase}_ns', 0) + p[f'{phase}_ns']
                totals[f'{phase}_bytes'] = totals.get(f'{phase}_bytes', 0) + p.get(f'{phase}_bytes', 0)
        totals['pl_busy_ns'] = round(totals['pl_cycles'] * 1e9 / clock_hz) if clock_hz and totals['pl_cycles'] else None
        programs.sort(key=lambda r: r['call_ns'], reverse=True)
        return {'clock_hz': clock_hz, 'totals': totals, 'programs': programs,
                'events_recorded': len(data['events']), 'events_dropped': data['events_dropped']}

    def report(self) -> dict[str, Any]:
        stages = {name: agg.as_dict() for name, agg in self.stages.items()}
        delegates = {device: self.delegate_report(device) for device in self.delegates}
        model_ns = sum(self.stages[name].total_ns for name in self.model_stages)
        delegate_ns = sum(d['totals']['call_ns'] for d in delegates.values())
        summary = {'model_ns': model_ns, 'delegate_ns': delegate_ns,
                   'other_ns': model_ns - delegate_ns if model_ns else None,
                   'delegate_share': round(delegate_ns / model_ns, 4) if model_ns else None,
                   'model_stages': sorted(self.model_stages)}
        return {'format': FORMAT, 'clock': 'CLOCK_MONOTONIC', 'meta': self.meta,
                'stages': stages, 'delegates': delegates, 'summary': summary}

    def table(self, top: int = 10) -> str:
        report = self.report()
        ms = lambda ns: f'{ns / 1e6:10.3f}' if ns is not None else f'{"-":>10}'
        lines = [f'{"stage":<18}{"count":>7}{"total ms":>11}{"mean ms":>11}{"min ms":>11}{"max ms":>11}']
        for name, s in report['stages'].items():
            lines.append(f'{name:<18}{s["count"]:>7}{ms(s["total_ns"])} {ms(s["mean_ns"])}'
                         f' {ms(s["min_ns"])} {ms(s["max_ns"])}')
        summary = report['summary']
        if summary['model_ns']:
            share = summary['delegate_share']
            lines += ['', f'package calls {ms(summary["model_ns"]).strip()} ms = delegates '
                          f'{ms(summary["delegate_ns"]).strip()} ms ({share:.1%}) + model.so and dispatch '
                          f'{ms(summary["other_ns"]).strip()} ms']
        for device, d in report['delegates'].items():
            t = d['totals']
            lines += ['', f'{device}: {t["calls"]} calls, {t["jobs"]} jobs, {t["polls"]} polls',
                      f'{"phase":<10}{"total ms":>11}{"share":>8}{"MB":>10}{"GB/s":>8}']
            for phase in PHASES:
                ns, nbytes = t[f'{phase}_ns'], t[f'{phase}_bytes']
                gbps = f'{nbytes / ns:8.3f}' if ns and nbytes else f'{"-":>8}'
                share = f'{ns / t["call_ns"]:8.1%}' if t['call_ns'] else f'{"-":>8}'
                lines.append(f'{phase:<10}{ms(ns)} {share}{nbytes / 1e6:10.2f}{gbps}')
            if t['pl_busy_ns'] is not None:
                lines.append(f'pl busy from cycles {ms(t["pl_busy_ns"]).strip()} ms; start and poll overhead '
                             f'{ms(t["pl_ns"] - t["pl_busy_ns"]).strip()} ms')
            if d['events_dropped']:
                lines.append(f'{d["events_dropped"]} timeline events dropped (raise the event capacity)')
            lines += ['', f'top {top} programs by time',
                      f'{"index":>6}{"m":>6}{"n":>6}{"k":>6}{"calls":>7}{"total ms":>11}'
                      f'{"pack_a":>9}{"stage_b":>9}{"pl":>9}{"unpack_c":>9}']
            for p in d['programs'][:top]:
                parts = ''.join(f'{p[f"{ph}_ns"] / p["call_ns"]:9.1%}' if p['call_ns'] else f'{"-":>9}'
                                for ph in PHASES)
                lines.append(f'{p["index"]:>6}{p["m"]:>6}{p["n"]:>6}{p["k"]:>6}{p["calls"]:>7}'
                             f'{ms(p["call_ns"])}{parts}')
        return '\n'.join(lines)

    def trace(self) -> dict[str, Any]:
        """Chrome trace events (``ph: X``, microseconds), as Kineto writes them."""
        origin = min([self.start_ns] + [b for _, b, _, _ in self.stage_events]
                     + [e[2] for d in self.delegates.values() for e in d['events']])
        us = lambda ns: (ns - origin) / 1e3
        events: list[dict[str, Any]] = [
            {'ph': 'M', 'name': 'process_name', 'pid': 0, 'args': {'name': 'pynqo runtime'}},
            {'ph': 'M', 'name': 'thread_name', 'pid': 0, 'tid': 0, 'args': {'name': 'engine'}}]
        for name, begin, end, args in self.stage_events:
            events.append({'ph': 'X', 'cat': 'user_annotation', 'name': name, 'pid': 0, 'tid': 0,
                           'ts': us(begin), 'dur': (end - begin) / 1e3, 'args': args})
        for tid, (device, d) in enumerate(self.delegates.items(), start=1):
            events.append({'ph': 'M', 'name': 'thread_name', 'pid': 0, 'tid': tid,
                           'args': {'name': f'{device} delegate'}})
            for index, phase, begin, end in d['events']:
                events.append({'ph': 'X', 'cat': 'kernel' if phase == 'pl' else 'cpu_op',
                               'name': phase, 'pid': 0, 'tid': tid, 'ts': us(begin),
                               'dur': (end - begin) / 1e3, 'args': {'program': index}})
        return {'traceEvents': events, 'displayTimeUnit': 'ms', 'otherData': {'format': FORMAT, **self.meta}}

    def write(self, directory: str | Path) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = [directory / 'profile.json']
        written[0].write_text(json.dumps(self.report(), indent=2) + '\n', encoding='utf-8')
        if self.events:
            written.append(directory / 'trace.json')
            written[1].write_text(json.dumps(self.trace()) + '\n', encoding='utf-8')
        return written


TRACE_EVENTS = 1 << 18   # per source: about 10 MB of events, enough for a few hundred tokens


def add_arguments(parser: Any) -> None:
    """The --profile options every task CLI accepts."""
    parser.add_argument('--profile', metavar='DIR', type=Path,
                        help='write profile.json (stage and delegate phase times) to DIR')
    parser.add_argument('--profile-trace', action='store_true',
                        help='with --profile, also write trace.json (Chrome trace events)')


def from_arguments(args: Any, exported: Any, **meta: Any) -> Profiler | None:
    """A profiler attached to the ExportedPackage when --profile was given, else None."""
    if not args.profile:
        return None
    profiler = Profiler(events=TRACE_EVENTS if args.profile_trace else 0, meta=meta)
    profiler.attach(exported)
    return profiler


def finish(profiler: Profiler | None, args: Any) -> None:
    """Detach, write the files, print the table to stderr."""
    if profiler is None:
        return
    import sys

    profiler.detach()
    written = profiler.write(args.profile)
    print(profiler.table(), file=sys.stderr)
    print('profile: ' + ', '.join(str(p) for p in written), file=sys.stderr)


class NullProfiler:
    """What engines use when no profiler is given: every stage is free."""

    _null = nullcontext()

    def stage(self, name: str, model: bool = False, **args: Any) -> nullcontext:
        return self._null


NULL_PROFILER = NullProfiler()
