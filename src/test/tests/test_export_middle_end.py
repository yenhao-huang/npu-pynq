"""npu-middle-end: the rewrites on the torch dialect, and lowering to linalg.
Skipped where torch or torch-mlir is missing, or the tool is not built
(src/export/pipeline.py's build_middle_end builds it)."""

import importlib.util
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from src.export import pipeline

HAVE_TOOLS = all(importlib.util.find_spec(m) for m in ('torch', 'torch_mlir', 'torchao'))


def tool_is_built():
    try:
        pipeline._linux(f'test -x {pipeline.middle_end_tool()}')
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


def run(net, emit):
    """``net`` through to_mlir and npu-middle-end; the text of what they wrote, in order."""
    import torch
    from src.export.frontend.to_mlir import to_mlir

    ep = torch.export.export(net, (torch.zeros(1, 3, 8, 8),), strict=False).run_decompositions()
    with tempfile.TemporaryDirectory() as d:
        imported = to_mlir({'forward': ep}, Path(d))
        out = pipeline.middle_end(imported, Path(d) / 'out.mlir', emit=emit)
        return imported.read_text(encoding='utf-8'), out.read_text(encoding='utf-8')


def code(text):
    return text.split('dialect_resources')[0]


def blob(text, name):
    data = re.search(rf'"?{re.escape(name)}"?: "0x([0-9A-Fa-f]+)"', text).group(1)
    return bytes.fromhex(data)[4:]


def floats(text, pattern):
    return np.frombuffer(blob(text, re.search(rf'dense_resource<(\S+{pattern})>', code(text)).group(1)), np.float32)


@unittest.skipUnless(HAVE_TOOLS and tool_is_built(), 'torch, torch-mlir and npu-middle-end are required')
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


@unittest.skipUnless(HAVE_TOOLS and tool_is_built(), 'torch, torch-mlir and npu-middle-end are required')
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


if __name__ == '__main__':
    unittest.main()
