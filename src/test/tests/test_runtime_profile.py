"""The runtime profiler: stage and phase aggregation, its report and trace, the
engines' stages, and libnpu_rt's phase counters on the host simulator."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

import numpy as np

from src.runtime.share.profile import FORMAT, NULL_PROFILER, Profiler


class FakeClock:
    """Advances by a fixed step on every read."""

    def __init__(self, step: int = 1000) -> None:
        self.now, self.step = 0, step

    def __call__(self) -> int:
        self.now += self.step
        return self.now


PROGRAM = {'index': 0, 'm': 4, 'n': 4, 'k': 8, 'segments': 1, 'calls': 2, 'call_ns_min': 400,
           'call_ns_max': 600, 'pack_a_ns': 100, 'stage_b_ns': 300, 'pl_ns': 500, 'unpack_c_ns': 100,
           'pl_cycles': 30, 'polls': 7, 'jobs': 2, 'pack_a_bytes': 64, 'stage_b_bytes': 64,
           'unpack_c_bytes': 128, 'runs': 2, 'instructions': 12}


class FakeBackend:
    clock_hz = 100_000_000

    def __init__(self) -> None:
        self.enabled = []

    def profile(self, enable, events=0):
        self.enabled.append((enable, events))


class FakePackage:
    def __init__(self) -> None:
        self.backends = {'npu': FakeBackend()}

    def profile(self, enable, events=0):
        self.backends['npu'].profile(enable, events)

    def phase_stats(self):
        idle = dict(PROGRAM, index=1, calls=0, pack_a_ns=0, stage_b_ns=0, pl_ns=0, unpack_c_ns=0)
        return {'npu': [PROGRAM, idle]}

    def events(self):
        return {'npu': ([(0, 'pack_a', 5000, 5100), (0, 'pl', 5100, 5600)], 3)}


class ProfilerTests(unittest.TestCase):
    def test_stage_aggregates(self) -> None:
        clock = FakeClock()
        prof = Profiler(clock=clock)
        for _ in range(3):
            with prof.stage('decode', model=True):
                clock.now += 4000          # each stage spans 5000 ns with the read step
        with prof.stage('sample'):
            pass
        stages = prof.report()['stages']
        self.assertEqual(stages['decode'], {'count': 3, 'total_ns': 15000, 'mean_ns': 5000.0,
                                            'min_ns': 5000, 'max_ns': 5000})
        self.assertEqual(stages['sample']['count'], 1)
        self.assertEqual(prof.report()['summary']['model_stages'], ['decode'])

    def test_a_raising_stage_is_still_recorded(self) -> None:
        prof = Profiler(clock=FakeClock())
        with self.assertRaises(ValueError), prof.stage('forward'):
            raise ValueError
        self.assertEqual(prof.report()['stages']['forward']['count'], 1)

    def test_delegate_phases_and_summary(self) -> None:
        clock = FakeClock()
        package = FakePackage()
        prof = Profiler(clock=clock, events=16)
        prof.attach(package)
        with prof.stage('decode', model=True):
            clock.now += 1000                  # 2000 ns, of which the delegate ran 1000
        prof.detach()
        self.assertEqual(package.backends['npu'].enabled, [(True, 16), (False, 0)])
        report = prof.report()
        self.assertEqual(report['format'], FORMAT)
        npu = report['delegates']['npu']
        self.assertEqual(len(npu['programs']), 1, 'programs never called are left out')
        row = npu['programs'][0]
        self.assertEqual(row['call_ns'], 1000)
        self.assertEqual(row['pl_busy_ns'], 300)       # 30 cycles at 100 MHz
        self.assertEqual(row['stage_b_gbps'], round(64 / 300, 4))
        self.assertIsNone(row['pl_gbps'])
        self.assertEqual(row['macs'], 4 * 4 * 8 * 2)
        self.assertEqual(npu['totals']['calls'], 2)
        self.assertEqual((npu['totals']['runs'], npu['totals']['instructions']), (2, 12))
        self.assertEqual((row['jobs_per_run'], row['instructions_per_run'], row['polls_per_run']), (1, 6, 3.5))
        self.assertEqual(row['overhead_ns_per_run'], 100)   # (500 ns pl - 300 ns busy) / 2 runs
        self.assertEqual(npu['events_dropped'], 3)
        summary = report['summary']
        self.assertEqual((summary['model_ns'], summary['delegate_ns'], summary['other_ns']), (2000, 1000, 1000))
        self.assertEqual(summary['delegate_share'], 0.5)
        table = prof.table()
        for text in ('decode', 'stage_b', 'pl busy from cycles', '3 timeline events dropped', '50.0%',
                     'npu: 2 calls, 2 runs, 2 jobs, 12 instructions, 7 polls',
                     'per run: 1.0 jobs, 6.0 instructions, 3.5 polls, 0.1 µs start and poll overhead'):
            self.assertIn(text, table)

    def test_trace_is_chrome_trace_events(self) -> None:
        clock = FakeClock()
        prof = Profiler(clock=clock, events=16, meta={'task': 'llm'})
        prof.attach(FakePackage())
        with prof.stage('decode', model=True, pos=7):
            pass
        prof.detach()
        trace = prof.trace()
        spans = [e for e in trace['traceEvents'] if e['ph'] == 'X']
        self.assertEqual([e['name'] for e in spans], ['decode', 'pack_a', 'pl'])
        self.assertEqual(spans[0]['args'], {'pos': 7})
        self.assertEqual(spans[2]['cat'], 'kernel')
        self.assertEqual(spans[2]['dur'], 0.5)            # 500 ns in microseconds
        self.assertGreaterEqual(min(e['ts'] for e in spans), 0)
        self.assertEqual(trace['otherData']['task'], 'llm')

    def test_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prof = Profiler(clock=FakeClock())
            with prof.stage('forward', model=True):
                pass
            self.assertEqual([p.name for p in prof.write(tmp)], ['profile.json'])
            prof.events = 4
            self.assertEqual([p.name for p in prof.write(tmp)], ['profile.json', 'trace.json'])
            self.assertEqual(json.loads((Path(tmp) / 'profile.json').read_text())['format'], FORMAT)

    def test_null_profiler_costs_nothing(self) -> None:
        with NULL_PROFILER.stage('decode', model=True, pos=1):
            pass
        self.assertIs(NULL_PROFILER.stage('a'), NULL_PROFILER.stage('b'))


class StubPackage:
    """Answers decode/prefill with logits that make token (pos % vocab) the argmax."""

    backends: dict = {}

    def __init__(self, vocab: int) -> None:
        self.vocab, self.calls = vocab, []

    def call(self, entry, inputs, outputs):
        self.calls.append(entry)
        logits = outputs[0]
        logits.fill(0)
        logits[0, -1, int(inputs[1][0]) % self.vocab] = 1

    def stats(self):
        return {}


class EngineStageTests(unittest.TestCase):
    def llm(self):
        from src.runtime.tasks.llm.engine import ExportedLLM

        llm = ExportedLLM.__new__(ExportedLLM)
        llm.package = StubPackage(vocab=11)
        llm.vocab, llm.chunk, llm.max_seq = 11, 4, 64
        llm.cache = [np.zeros((1,), np.float32)]
        llm.spare = [np.zeros((1,), np.float32)]
        llm.logits = np.zeros((1, 1, 11), np.float32)
        llm.tokenizer, llm.eos, llm.pos = None, set(), 0
        return llm

    def test_llm_stages(self) -> None:
        prof = Profiler(clock=FakeClock())
        ids, _, _ = self.llm().generate([1, 2, 3, 4, 5], max_new_tokens=3, profiler=prof)
        stages = prof.report()['stages']
        self.assertEqual(stages['prefill']['count'], 2)          # 5 tokens in chunks of 4
        self.assertEqual(stages['decode']['count'], 2)           # the last token is not decoded
        self.assertEqual(stages['sample']['count'], 3)
        self.assertEqual(prof.report()['summary']['model_stages'], ['decode', 'prefill'])
        self.assertEqual(ids, self.llm().generate([1, 2, 3, 4, 5], max_new_tokens=3)[0],
                         'profiling does not change the output')


def write_package(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """One NPU program, C[4x4] = A[4x8] @ B[8x4], in libnpu_rt's package format."""
    rng = np.random.default_rng(1)
    a = rng.integers(-128, 128, (4, 8), dtype=np.int8)
    b = rng.integers(-128, 128, (8, 4), dtype=np.int8)
    words = [0x10 << 56 | 4 << 24 | 4 << 16 | 8,   # SHAPE m n k
             0x11 << 56 | 0,                         # ADDR_A
             0x12 << 56 | 0,                         # ADDR_B, relocated to the stage window
             0x13 << 56 | 32,                        # ADDR_C = c_base
             0x20 << 56,                             # RUN
             0x01 << 56]                             # END; ISA_INSTRUCTIONS counts all 6 words
    (root / 'programs.bin').write_bytes(struct.pack(f'<{len(words)}Q', *words))
    (root / 'programs.txt').write_text('npu-programs 1 16 16 256 6 1\n'
                                       'task 0 4 4 8 16 16 256 32 32 256 1\n'
                                       'seg 0 6 0 32 1 2\n')
    (root / 'weights.bin').write_bytes(b.tobytes())
    return a, b


@unittest.skipUnless(sys.platform.startswith('linux') and (shutil.which('c++') or shutil.which('cmake')),
                     'libnpu_rt builds on Linux with a C++ compiler')
class NpuPhaseTests(unittest.TestCase):
    def test_phase_counters_on_the_simulator(self) -> None:
        from src.runtime.delegate.npu import NpuBackend

        with tempfile.TemporaryDirectory() as tmp:
            a, b = write_package(Path(tmp))
            backend = NpuBackend(Path(tmp))
            lib = backend.lib
            lib.npu_rt_execute.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
            c = np.zeros((1, 4, 4), np.int32)
            buffers = (ctypes.c_void_p * 2)(a.ctypes.data, c.ctypes.data)
            try:
                self.assertEqual(lib.npu_rt_execute(0, buffers, 2), 0)
                self.assertEqual(backend.phase_stats()[0]['calls'], 0, 'off by default')
                backend.profile(True, events=5)
                for _ in range(3):
                    c.fill(0)
                    self.assertEqual(lib.npu_rt_execute(0, buffers, 2), 0)
                    np.testing.assert_array_equal(c[0], a.astype(np.int32) @ b.astype(np.int32))
                [row] = backend.phase_stats()
                self.assertEqual({k: row[k] for k in ('m', 'n', 'k', 'segments', 'calls', 'runs', 'jobs',
                                                      'instructions')},
                                 {'m': 4, 'n': 4, 'k': 8, 'segments': 1, 'calls': 3, 'runs': 3, 'jobs': 3,
                                  'instructions': 3 * 6})
                self.assertEqual((row['pack_a_bytes'], row['stage_b_bytes'], row['unpack_c_bytes']),
                                 (3 * 32, 3 * 32, 3 * 64))
                phases = row['pack_a_ns'] + row['stage_b_ns'] + row['pl_ns'] + row['unpack_c_ns']
                self.assertLessEqual(row['call_ns_min'] * 3, phases)
                self.assertGreaterEqual(row['call_ns_max'] * 3, phases)
                events, dropped = backend.events()
                self.assertEqual([e[1] for e in events], ['pack_a', 'stage_b', 'pl', 'unpack_c', 'pack_a'])
                self.assertEqual(dropped, 3 * 4 - 5)
                self.assertTrue(all(begin <= end for _, _, begin, end in events))
                self.assertEqual(backend.stats()['calls'], 4)
                backend.profile(False)
                self.assertEqual(backend.phase_stats()[0]['calls'], 0)
            finally:
                backend.close()


if __name__ == '__main__':
    unittest.main()
