"""Kernel-level CPU profiling: pynqo-middle-end numbers every CPU kernel
(pynqo.kernel_id), pynqo-backend --profile-kernels wraps each in
__prof_enter / __prof_exit and writes kernels.json, and the runtime times them
(src/runtime/share/kernels.py). Skipped where torch, torch-mlir or the tools
are missing; the runtime tests also need Linux."""

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

from src.export import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_export_middle_end import HAVE_TOOLS, MLP_INPUT, code, lower, mlp, tool_is_built  # noqa: E402

KERNEL_ID = re.compile(r'pynqo\.kernel_id = (\d+) : i32')
TOOLS = HAVE_TOOLS and tool_is_built() and tool_is_built(pipeline.backend_tool())


def backend(model, out, devices='cpu', extra=''):
    pipeline._linux(f"{pipeline.backend_tool()} '{pipeline._linux_path(model)}' "
                    f"--out '{pipeline._linux_path(out)}' --devices {devices} --cpus native {extra}")
    compiled = json.loads((Path(out) / 'compile.json').read_text())
    (model_dir,) = compiled['libraries']
    return compiled, Path(out) / model_dir


@unittest.skipUnless(HAVE_TOOLS and tool_is_built(), 'torch, torch-mlir and pynqo-middle-end are required')
class KernelIdTest(unittest.TestCase):
    def test_kernels_are_numbered_in_program_order(self):
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, shape=MLP_INPUT)
            text = code(model.read_text(encoding='utf-8'))
        ids = [int(i) for i in KERNEL_ID.findall(text)]
        self.assertGreater(len(ids), 3)
        self.assertEqual(ids, list(range(len(ids))))
        self.assertRegex(text, r'linalg\.matmul \{pynqo\.kernel_id = \d+ : i32\}')
        # Views move no data and are not kernels.
        for line in text.splitlines():
            if re.search(r'tensor\.(expand_shape|collapse_shape|extract_slice) ', line):
                self.assertNotIn('kernel_id', line)

    def test_delegate_calls_are_kernels_and_device_modules_are_not_numbered(self):
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, devices=('cpu', 'npu'), shape=MLP_INPUT)
            text = code(model.read_text(encoding='utf-8'))
            device = (pipeline.delegates_dir(model) / 'npu.mlir').read_text(encoding='utf-8')
        self.assertEqual(len(re.findall(r'call @__delegate_\d+\(.*\{pynqo\.kernel_id = \d+ : i32\}', text)), 2)
        self.assertNotIn('kernel_id', device)


@unittest.skipUnless(TOOLS, 'torch, torch-mlir, pynqo-middle-end and pynqo-backend are required')
class KernelProbeTest(unittest.TestCase):
    def test_without_the_flag_kernel_ids_change_no_code(self):
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, shape=MLP_INPUT)
            text = model.read_text(encoding='utf-8')
            stripped = Path(d) / 'stripped.mlir'
            stripped.write_text(re.sub(r'(, )?pynqo\.kernel_id = \d+ : i32', '', text)
                                .replace(' attrs =  {}', '').replace(' {}', ''), encoding='utf-8')
            self.assertNotIn('kernel_id', stripped.read_text(encoding='utf-8'))
            compiled, plain = backend(model, Path(d) / 'plain')
            _, bare = backend(stripped, Path(d) / 'bare')
            self.assertNotIn('kernels', compiled)
            self.assertFalse((plain / 'kernels.json').exists())
            so = (plain / 'model.so').read_bytes()
            self.assertNotIn(b'__prof_enter', so)
            self.assertEqual(so, (bare / 'model.so').read_bytes())

    def test_every_kernel_is_probed_and_described(self):
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, devices=('cpu', 'npu'), shape=MLP_INPUT)
            ids = [int(i) for i in KERNEL_ID.findall(code(model.read_text(encoding='utf-8')))]
            compiled, lib = backend(model, Path(d) / 'pkg', 'cpu,npu', '--profile-kernels')
            self.assertEqual(compiled['kernels'], {lib.name: f'{lib.name}/kernels.json'})
            table = json.loads((lib / 'kernels.json').read_text())
            self.assertIn(b'__prof_enter', (lib / 'model.so').read_bytes())
        self.assertEqual(table['format'], 'pynqo-kernels/1')
        kernels = table['kernels']
        self.assertEqual([k['id'] for k in kernels], list(range(len(kernels))))
        # Every number the middle end gave is there; copies added by
        # bufferization come after them.
        self.assertEqual([k['id'] for k in kernels if k['kind'] != 'copy'], ids)
        self.assertTrue(all(k['id'] > max(ids) for k in kernels if k['kind'] == 'copy'))
        delegates = [k for k in kernels if k['kind'] == 'delegate']
        self.assertEqual([k['delegate'] for k in delegates], [0, 1])
        self.assertTrue(all(k['function'] == 'forward' for k in kernels))


@unittest.skipUnless(TOOLS and sys.platform.startswith('linux'), 'the tools and Linux are required')
class KernelTimingTest(unittest.TestCase):
    """A probed package computes what the plain one does and times every kernel."""

    def package(self, model, out, extra=''):
        from src.runtime.share.loader import ExportedPackage

        compiled, _ = backend(model, out, 'cpu', extra)
        manifest = {**compiled, 'format': pipeline.PACKAGE_FORMAT}
        (Path(out) / 'manifest.json').write_text(json.dumps(manifest))
        return ExportedPackage(out)

    def test_probed_kernels_are_timed(self):
        from src.runtime.share.profile import Profiler

        x = np.random.default_rng(1).standard_normal(MLP_INPUT).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, shape=MLP_INPUT)
            plain = self.package(model, Path(d) / 'plain')
            probed = self.package(model, Path(d) / 'probed', '--profile-kernels')
            self.assertIsNone(plain.kernel_stats())
            expected, got = np.zeros((4, 8), np.float32), np.zeros((4, 8), np.float32)
            plain.call('forward', [x], [expected])
            profiler = Profiler(events=64)
            profiler.attach(probed)
            for _ in range(3):
                with profiler.stage('forward', model=True):
                    probed.call('forward', [x], [got])
            profiler.detach()
            np.testing.assert_array_equal(got, expected)
            table = probed.kernels.table
            report = profiler.report()['kernels']
            self.assertEqual(sorted(k['id'] for k in report['kernels']), [k['id'] for k in table])
            self.assertTrue(all(k['calls'] == 3 for k in report['kernels']))
            self.assertLessEqual(report['probed_ns'], profiler.stages['forward'].total_ns)
            self.assertEqual(report['events_recorded'], min(64, 3 * len(table)))
            self.assertIn('top 10 CPU kernels by time', profiler.table())
            # Groups sum the CPU kernels (delegates aside) without losing any.
            cpu = [k for k in report['kernels'] if k['kind'] != 'delegate']
            self.assertEqual(sum(g['kernels'] for g in report['groups']), len(cpu))
            self.assertEqual(sum(g['total_ns'] for g in report['groups']), report['cpu_ns'])
            spans = [e for e in profiler.trace()['traceEvents'] if e.get('args', {}).get('kernel') is not None]
            self.assertEqual(len(spans), report['events_recorded'])
            # Profiling off: the probes return at once and count nothing.
            probed.profile(True)
            probed.profile(False)
            probed.call('forward', [x], [got])
            self.assertEqual(probed.kernel_stats(), [])


if __name__ == '__main__':
    unittest.main()
