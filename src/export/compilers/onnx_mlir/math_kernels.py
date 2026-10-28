"""Fixed FP32 arithmetic for the pinned quantized graph's softmax boundary.

The range reduction and polynomial coefficients follow Eigen/MLAS, as used
by ONNX Runtime v1.20.1 core/mlas/lib/compute.cpp. The generated computation
has no ORT calls or CPU-specific vector instructions.
"""

from src.export.compilers.mlir.builder import FuncBuilder, Region

SOFTMAX_EXP = 'onnx_softmax_exp_f32'


def softmax_exp_text():
    """Exp on non-positive softmax arguments with explicit fused rounding."""
    function = FuncBuilder(SOFTMAX_EXP, [('x', 'f32')], ['f32'], c_interface=False, private=True)
    r = Region(function, [])

    def binary(op, left, right):
        return r.binary('arith.' + op, left, right, fast=False)

    def fma(a, b, c):
        return r.op(f'math.fma {a}, {b}, {c} : f32', 'f32')

    x = r.maxf(function.arg('x'), r.const(-88.3762626647949, 'f32'))
    bias = r.const(12582912.0, 'f32')
    biased = fma(x, r.const(1.44269504088896341, 'f32'), bias)
    exponent = binary('subf', biased, bias)
    reduced = fma(exponent, r.const(-6.93145752e-1, 'f32'), x)
    reduced = fma(exponent, r.const(-1.42860677e-6, 'f32'), reduced)
    bits = r.cast('arith.bitcast', biased, 'i32')
    bits = binary('shli', bits, r.const(23, 'i32'))
    bits = r.addi(bits, r.const(0x3f800000, 'i32'))
    scale = r.cast('arith.bitcast', bits, 'f32')
    coefficients = [float.fromhex(value) for value in
        ('0x1.694000p-10', '0x1.125edcp-7', '0x1.555b5ap-5', '0x1.555450p-3',
         '0x1.fffff6p-2', '0x1.000000p+0', '0x1.000000p+0')]
    result = r.const(coefficients[0], 'f32')
    for coefficient in coefficients[1:]:
        result = fma(result, reduced, r.const(coefficient, 'f32'))
    result = binary('mulf', result, scale)
    function.lines.extend(r.lines)
    function.emit(f'return {result} : f32')
    return function.render()
