"""pynqo-middle-end: the rewrites on the torch dialect, lowering to linalg,
device assignment, partitioning into delegate calls and device modules, and
fusion; and pynqo-backend's side of that contract. Skipped where torch or torch-mlir is missing, or a tool is not
built (src/export/pipeline.py's build_middle_end and build_backend build them)."""

import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from src.export import pipeline

HAVE_TOOLS = all(importlib.util.find_spec(m) for m in ('torch', 'torch_mlir', 'torchao'))


def tool_is_built(tool=None):
    try:
        pipeline._linux(f'test -x {tool or pipeline.middle_end_tool()}')
    except (OSError, RuntimeError):
        return False
    return True


def conv_bn(bias=True):
    import torch
    from torch import nn

    class ConvBn(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=bias)
            self.bn = nn.BatchNorm2d(4)

        def forward(self, x):
            return torch.relu(self.bn(self.conv(x)))

    torch.manual_seed(3)
    net = ConvBn().eval().requires_grad_(False)
    net.bn.running_mean.uniform_(-1, 1)
    net.bn.running_var.uniform_(0.5, 2)
    return net


def w8a8(net):
    from src.export.frontend.quantization.torchao.config import quantize

    return quantize(net, 'w8a8')


def mlp():
    """Two INT8 linear layers on a [4, 16] input: two NPU contractions, in module order."""
    import torch
    from torch import nn

    torch.manual_seed(5)
    return w8a8(nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval())


def lower(net, d, emit='linalg', devices=('cpu',), shape=(1, 3, 8, 8)):
    """``net`` through to_mlir and pynqo-middle-end into directory ``d``; both files' paths."""
    import torch
    from src.export.frontend.to_mlir import to_mlir

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ep = torch.export.export(net, (torch.zeros(*shape),), strict=False).run_decompositions()
    imported = to_mlir({'forward': ep}, d)
    return imported, pipeline.middle_end(imported, d / 'out.mlir', emit=emit, devices=devices)


def run(net, emit, devices=('cpu',), shape=(1, 3, 8, 8)):
    """``net`` through to_mlir and pynqo-middle-end; the text of what they wrote, in order."""
    imported, out, _ = run_with_delegates(net, emit, devices, shape)
    return imported, out


def run_with_delegates(net, emit, devices=('cpu',), shape=(1, 3, 8, 8)):
    """Like run, plus device -> the text of its module (ir/delegates/<device>.mlir)."""
    with tempfile.TemporaryDirectory() as d:
        imported, out = lower(net, d, emit, devices, shape)
        modules = {p.stem: p.read_text(encoding='utf-8')
                   for p in sorted(pipeline.delegates_dir(out).glob('*.mlir'))}
        return imported.read_text(encoding='utf-8'), out.read_text(encoding='utf-8'), modules


def functions(text):
    """name -> (signature line, body) for every func.func in ``text``, in order.
    A declaration (no body) maps to its line and an empty body."""
    text = code(text)
    found = {}
    for m in re.finditer(r'^  func\.func (?:private )?@(\w+)\(.*$', text, re.M):
        body = '' if not m.group(0).endswith('{') else text[m.end():text.index('\n  }\n', m.end())]
        found[m.group(1)] = (m.group(0), body)
    return found


def code(text):
    return text.split('dialect_resources')[0]


def blob(text, name):
    data = re.search(rf'"?{re.escape(name)}"?: "0x([0-9A-Fa-f]+)"', text).group(1)
    return bytes.fromhex(data)[4:]


def floats(text, pattern):
    return np.frombuffer(blob(text, re.search(rf'dense_resource<(\S+{pattern})>', code(text)).group(1)), np.float32)


@unittest.skipUnless(HAVE_TOOLS and tool_is_built(), 'torch, torch-mlir and pynqo-middle-end are required')
class FuseConvBatchNormTest(unittest.TestCase):
    def test_weights_are_pytorchs_fused_weights(self):
        from torch.nn.utils.fusion import fuse_conv_bn_weights

        for bias in (True, False):
            with self.subTest(bias=bias):
                net = conv_bn(bias)
                _, text = run(net, 'torch')
                self.assertNotIn('batch_norm', code(text))
                weight, fused_bias = fuse_conv_bn_weights(
                    net.conv.weight, net.conv.bias, net.bn.running_mean, net.bn.running_var, net.bn.eps,
                    net.bn.weight, net.bn.bias)
                np.testing.assert_array_equal(floats(text, '_fused_weight'), weight.detach().numpy().ravel())
                np.testing.assert_array_equal(floats(text, '_fused_bias'), fused_bias.detach().numpy().ravel())

    def test_w8a8_rescale_takes_the_batchnorm(self):
        from torch.nn.utils.fusion import fuse_conv_bn_weights

        for bias in (True, False):
            with self.subTest(bias=bias):
                q = w8a8(conv_bn(bias))
                conv, bn = q.conv, q.bn
                imported, text = run(q, 'torch')
                self.assertNotIn('batch_norm', code(text))
                # The weights the NPU reads are untouched.
                name = 'torch_tensor_27_4_torch.int8'
                self.assertEqual(blob(text, name), blob(imported, name))
                scale, fused_bias = fuse_conv_bn_weights(conv.scale.unsqueeze(1), conv.bias, bn.running_mean,
                                                         bn.running_var, bn.eps, bn.weight, bn.bias)
                np.testing.assert_array_equal(floats(text, '_fused_scale'), scale.detach().numpy().ravel())
                np.testing.assert_array_equal(floats(text, '_fused_bias'), fused_bias.detach().numpy().ravel())

    def test_a_convolution_output_used_elsewhere_is_left_alone(self):
        import torch

        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pair = conv_bn()

            def forward(self, x):
                y = self.pair.conv(x)
                return self.pair.bn(y) + y

        _, text = run(Shared().eval(), 'torch')
        self.assertIn('batch_norm', code(text))

    def test_a_batchnorm_after_something_else_is_left_alone(self):
        import torch

        _, text = run(torch.nn.Sequential(torch.nn.Identity(), torch.nn.BatchNorm2d(3)).eval(), 'torch')
        self.assertIn('batch_norm', code(text))


@unittest.skipUnless(HAVE_TOOLS and tool_is_built(), 'torch, torch-mlir and pynqo-middle-end are required')
class LoweringTest(unittest.TestCase):
    def test_float_model_lowers_to_upstream_dialects(self):
        _, text = run(conv_bn(), 'linalg')
        self.assertIn('linalg.conv_2d_nchw_fchw', text)
        self.assertNotIn('torch.', re.sub(r'torch_tensor\S*', '', text))

    def test_w8a8_contraction_stays_an_int8_matmul(self):
        _, text = run(w8a8(conv_bn(False)), 'linalg')
        self.assertRegex(text, r'linalg\.matmul ins\(.*xi8>, tensor<\S*xi8>\) outs\(.*xi32>\)')
        self.assertNotIn('torch.', re.sub(r'torch_tensor\S*', '', text))

    def test_any_dim_lowers(self):
        import torch

        class Any(torch.nn.Module):
            def forward(self, x):
                return (x > 0).any(dim=1, keepdim=True).float() + x.sum(dim=1, keepdim=True)

        _, text = run(Any().eval(), 'linalg')
        self.assertNotIn('torch.', re.sub(r'torch_tensor\S*', '', text))

    def test_elementwise_chain_is_fused(self):
        # canonicalize, cse and linalg-fuse-elementwise-ops run in the middle end,
        # once, so model.mlir is the fused graph every CPU target compiles.
        import torch

        class Chain(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x * 2.0 + 1.0).neg()

        _, text = run(Chain().eval(), 'linalg')
        self.assertEqual(code(text).count('linalg.generic'), 1)


MLP_INPUT = (4, 16)
# What the middle end promises pynqo-backend: a bodiless declaration in the CPU
# graph, and the subgraph with the same id and type in the device's module.
DECLARATION = (r'func\.func private @__delegate_(\d+)\(tensor<(\w+)xi8>\) -> tensor<(\w+)xi32> '
               r'attributes \{pynqo\.delegate_id = (\d+) : i64, pynqo\.device = "npu"\}$')
SUBGRAPH = (r'func\.func @npu_(\d+)\(%arg0: tensor<(\w+)xi8>\) -> tensor<(\w+)xi32> '
            r'attributes \{pynqo\.delegate_id = (\d+) : i64\} \{$')


@unittest.skipUnless(HAVE_TOOLS and tool_is_built(), 'torch, torch-mlir and pynqo-middle-end are required')
class DevicePartitionTest(unittest.TestCase):
    def test_cpu_alone_assigns_nothing(self):
        _, text, modules = run_with_delegates(mlp(), 'linalg', shape=MLP_INPUT)
        self.assertIn('module attributes {pynqo.devices = ["cpu"]', text)
        self.assertNotIn('pynqo.device ', text)
        self.assertNotIn('@__delegate_', text)
        self.assertEqual(modules, {})
        self.assertEqual(code(text).count('linalg.matmul'), 2)

    def test_each_npu_contraction_becomes_one_delegate_call(self):
        _, text, modules = run_with_delegates(mlp(), 'linalg', ('cpu', 'npu'), MLP_INPUT)
        self.assertIn('module attributes {pynqo.devices = ["cpu", "npu"]', text)
        fns = functions(text)
        # The CPU graph keeps one bodiless declaration per delegate; ids follow module order.
        declared = [re.search(DECLARATION, h) for name, (h, _) in fns.items() if name.startswith('__delegate_')]
        self.assertTrue(declared and all(declared), list(fns))
        self.assertEqual([(h.group(1), h.group(4)) for h in declared], [('0', '0'), ('1', '1')])
        self.assertEqual([(h.group(2), h.group(3)) for h in declared], [('4x16', '4x32'), ('4x32', '4x8')])
        self.assertTrue(all(body == '' for name, (_, body) in fns.items() if name.startswith('__delegate_')))
        # The caller keeps the CPU work and calls the delegates in order; nothing
        # about how the device computes the result is left in the CPU graph.
        _, forward = fns['forward']
        self.assertNotIn('linalg.matmul', code(text))
        self.assertNotIn('@npu_', text)
        self.assertEqual(re.findall(r'call @(__delegate_\d+)\(', forward), ['__delegate_0', '__delegate_1'])
        # The subgraphs live in the device's module, with the same ids and types.
        self.assertEqual(list(modules), ['npu'])
        npu = modules['npu']
        self.assertIn('module @npu attributes {pynqo.device = "npu"}', npu)
        subgraphs = functions(npu)
        headers = [re.search(SUBGRAPH, h) for h, _ in subgraphs.values()]
        self.assertTrue(headers and all(headers), list(subgraphs))
        self.assertEqual([(h.group(1), h.group(4)) for h in headers], [('0', '0'), ('1', '1')])
        self.assertEqual([(h.group(2), h.group(3)) for h in headers], [('4x16', '4x32'), ('4x32', '4x8')])
        # Each body holds its weight and accumulator initialization: activation in, result out.
        for name in ('npu_0', 'npu_1'):
            _, body = subgraphs[name]
            self.assertRegex(body, r'arith\.constant dense_resource<torch_tensor_\w+_torch\.int8> : tensor<\w+xi8>')
            self.assertRegex(body, r'linalg\.fill ins\(%c0_i32 : i32\)')
            self.assertRegex(body, r'linalg\.matmul ins\(%arg0, %cst : tensor<\w+xi8>, tensor<\w+xi8>\) '
                                   r'outs\(%1 : tensor<\w+xi32>\)')
            self.assertNotIn('pynqo.device', body)
            self.assertNotIn('call ', body)

    def test_a_w8a8_convolution_contraction_moves_whole(self):
        _, text, modules = run_with_delegates(w8a8(conv_bn(False)), 'linalg', ('cpu', 'npu'))
        fns = functions(text)
        self.assertEqual([n for n in fns if n.startswith('__delegate_')], ['__delegate_0'])
        self.assertEqual(list(functions(modules['npu'])), ['npu_0'])
        self.assertRegex(functions(modules['npu'])['npu_0'][1],
                         r'linalg\.matmul ins\(.*xi8>, tensor<\S*xi8>\) outs\(.*xi32>\)')
        self.assertNotIn('linalg.matmul', fns['forward'][1])

    def test_float_work_stays_on_the_cpu(self):
        _, text, modules = run_with_delegates(conv_bn(), 'linalg', ('cpu', 'npu'))
        self.assertIn('module attributes {pynqo.devices = ["cpu", "npu"]', text)
        self.assertNotIn('@__delegate_', text)
        self.assertNotIn('pynqo.device ', text)
        self.assertEqual(modules, {})

    def test_unknown_device_lists_fail(self):
        for devices in (('cpu', 'gpu'), ('npu',), ('cpu', 'npu', 'npu')):
            with self.subTest(devices=devices), self.assertRaises(RuntimeError):
                run(mlp(), 'linalg', devices, MLP_INPUT)


@unittest.skipUnless(HAVE_TOOLS and tool_is_built() and tool_is_built(pipeline.backend_tool()),
                     'torch, torch-mlir, pynqo-middle-end and pynqo-backend are required')
class BackendContractTest(unittest.TestCase):
    """pynqo-backend keeps the middle end's delegate ids and refuses another device list."""

    def backend(self, model, out, devices, extra=''):
        pipeline._linux(f"{pipeline.backend_tool()} '{pipeline._linux_path(model)}' "
                        f"--out '{pipeline._linux_path(out)}' --devices {devices} --cpus native --keep-ir {extra}")

    def test_ids_are_kept_and_device_modules_are_compiled(self):
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, devices=('cpu', 'npu'), shape=MLP_INPUT)
            self.backend(model, Path(d) / 'pkg', 'cpu,npu')
            compiled = json.loads((Path(d) / 'pkg' / 'compile.json').read_text())
            self.assertEqual(compiled['format'], 'pynqo-backend/2')
            table = compiled['delegates']
            self.assertEqual([(e['id'], e['device'], e['index']) for e in table], [(0, 'npu', 0), (1, 'npu', 1)])
            # Bytes in and out are the subgraph's argument and result: M x K INT8, M x N INT32.
            self.assertEqual([(e['input_bytes'], e['output_bytes']) for e in table], [(64, 512), (128, 128)])
            cpu = code((Path(d) / 'pkg' / 'cpu.mlir').read_text())
            self.assertNotIn('@npu_', cpu)
            self.assertNotRegex(cpu, r'@__delegate_\d')
            self.assertNotIn('pynqo.', cpu)
            self.assertEqual(len(re.findall(r'call @__delegate__\(', cpu)), 2)
            # How the NPU computes the result (K chunks) is not CPU code.
            self.assertNotIn('linalg.reduce', cpu)
            self.assertNotIn('linalg.matmul', cpu)

    def test_k_chunks_change_no_cpu_code_and_no_output_size(self):
        # MAX_K 8 splits K = 16 and K = 32 into 2 and 4 chunks; the delegate
        # table and the CPU graph stay those of one chunk.
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, devices=('cpu', 'npu'), shape=MLP_INPUT)
            self.backend(model, Path(d) / 'pkg', 'cpu,npu', '--max-k 8')
            table = json.loads((Path(d) / 'pkg' / 'compile.json').read_text())['delegates']
            self.assertEqual([(e['input_bytes'], e['output_bytes']) for e in table], [(64, 512), (128, 128)])
            tasks = json.loads((Path(d) / 'pkg' / 'programs.json').read_text())['tasks']
            self.assertEqual([(t['k'], t['tk']) for t in tasks], [(16, 8), (32, 8)])
            self.assertNotIn('linalg.reduce', code((Path(d) / 'pkg' / 'cpu.mlir').read_text()))

    def test_a_missing_subgraph_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _, model = lower(mlp(), d, devices=('cpu', 'npu'), shape=MLP_INPUT)
            npu = pipeline.delegates_dir(model) / 'npu.mlir'
            text = npu.read_text(encoding='utf-8')
            start = text.index('  func.func @npu_1(')
            npu.write_text(text[:start] + text[text.index('\n  }\n', start) + 5:], encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, r'has no subgraph for delegate 1'):
                self.backend(model, Path(d) / 'pkg', 'cpu,npu')

    def test_a_model_assigned_for_other_devices_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _, npu = lower(mlp(), Path(d) / 'npu', devices=('cpu', 'npu'), shape=MLP_INPUT)
            with self.assertRaisesRegex(RuntimeError, r'assigned for devices \[cpu,npu\] but --devices is \[cpu\]'):
                self.backend(npu, Path(d) / 'pkg', 'cpu')
            _, cpu = lower(mlp(), Path(d) / 'cpu', shape=MLP_INPUT)
            with self.assertRaisesRegex(RuntimeError, r'assigned for devices \[cpu\] but --devices is \[cpu,npu\]'):
                self.backend(cpu, Path(d) / 'pkg', 'cpu,npu')


if __name__ == '__main__':
    unittest.main()
