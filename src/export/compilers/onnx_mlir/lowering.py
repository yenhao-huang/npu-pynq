"""Lower imported ONNX operator semantics to the existing CPU MLIR dialects,
including the INT8 projections that keep the source's quantization
(projection) and calls into native kernels (native)."""

from __future__ import annotations

import numpy as np

from src.export.compilers.mlir.builder import identity_map, map_of, memref_type


def _float(region, operation, left, right):
    return region.binary('arith.' + operation, left, right, fast=False)


def rms_norm(builder, value, gamma, epsilon=1e-5):
    """Last-axis FP32 RMS normalization with ordered accumulation."""
    if value.rank != 2 or gamma.shape != (value.shape[1],):
        raise ValueError('RMS normalization requires rank-2 input and matching gamma')
    rows, width = value.shape

    def reduce(region):
        x, total = region.args
        return [_float(region, 'addf', total, _float(region, 'mulf', x, x))]

    sums = builder.generic([value], [builder.fill(0.0, [rows], 'f32')],
        [identity_map(2), map_of(2, ['d0'])], ['parallel', 'reduction'], reduce)[0]

    def denominator(region):
        mean = _float(region, 'divf', region.args[0], region.const(width, 'f32'))
        shifted = _float(region, 'addf', mean, region.const(epsilon, 'f32'))
        return region.op(f'math.sqrt {shifted} : f32', 'f32')

    denominators = builder.elementwise([sums], 'f32', denominator)
    return builder.elementwise([value, denominators, gamma], 'f32', lambda region:
        _float(region, 'mulf', _float(region, 'divf', region.args[0], region.args[1]), region.args[2]),
        maps=[identity_map(2), map_of(2, ['d0']), map_of(2, ['d1'])])


def skip_norm(builder, left, right, gamma, epsilon=1e-5):
    residual = builder.elementwise([left, right], 'f32', lambda region:
        _float(region, 'addf', *region.args[:2]))
    return rms_norm(builder, residual, gamma, epsilon), residual


def dynamic_quantize(builder, value):
    """ONNX DynamicQuantizeLinear: one asymmetric UINT8 range for the tensor."""
    rank = value.rank
    scalar = map_of(rank, ['0'])

    def extrema(region):
        x, lo, hi = region.args
        return [region.minf(x, lo), region.maxf(x, hi)]

    lo, hi = builder.generic([value], [builder.fill(0.0, [1], 'f32'),
        builder.fill(0.0, [1], 'f32')], [identity_map(rank), scalar, scalar],
        ['reduction'] * rank, extrema)

    def scale_body(region):
        span = _float(region, 'subf', region.args[1], region.args[0])
        scale = _float(region, 'divf', span, region.const(255.0, 'f32'))
        return region.select(region.cmpf('oeq', span, region.const(0.0, 'f32')),
                             region.const(1.0, 'f32'), scale)

    scale = builder.elementwise([lo, hi], 'f32', scale_body)

    def zero_body(region):
        minimum, scale = region.args[:2]
        zero = _float(region, 'divf', region.op(f'arith.negf {minimum} : f32', 'f32'), scale)
        zero = region.roundeven(region.minf(region.maxf(zero, region.const(0.0, 'f32')),
                                           region.const(255.0, 'f32')))
        return region.cast('arith.fptoui', zero, 'i8')

    zero = builder.elementwise([lo, scale], 'i8', zero_body)

    def quant_body(region):
        x, scale, zero = region.args[:3]
        scaled = region.roundeven(_float(region, 'divf', x, scale))
        zero = region.cast('arith.uitofp', zero, 'f32')
        q = _float(region, 'addf', scaled, zero)
        q = region.minf(region.maxf(q, region.const(0.0, 'f32')), region.const(255.0, 'f32'))
        return region.cast('arith.fptoui', q, 'i8')

    quantized = builder.elementwise([value, scale, zero], 'i8', quant_body,
        maps=[identity_map(rank), scalar, scalar])
    return quantized, scale, zero


def rotary(builder, value, cos, sin, position, head_dim=64):
    """Noninterleaved full-width ONNX RotaryEmbedding for static token profiles."""
    if head_dim % 2 or value.rank != 2 or value.shape[1] % head_dim:
        raise ValueError('Rotary input must contain complete even-sized heads')
    half = head_dim // 2
    if cos.shape != sin.shape or cos.shape[1] != half:
        raise ValueError('Rotary caches must have half a head per position')

    def body(region):
        row, column = region.index(0), region.index(1)
        offset = region.op(f'arith.remui {column}, {region.const(head_dim, "index")} : index', 'index')
        first = region.cmpi('ult', offset, region.const(half, 'index'))
        paired = region.select(first, region.addi(column, region.const(half, 'index')),
                               region.subi(column, region.const(half, 'index')))
        index = region.op(f'arith.remui {column}, {region.const(half, "index")} : index', 'index')
        pos = region.addi(position, row)
        x, other = region.extract(value, [row, column]), region.extract(value, [row, paired])
        product = _float(region, 'mulf', other, region.extract(sin, [pos, index]))
        product = region.select(first, region.op(f'arith.negf {product} : f32', 'f32'), product)
        return [_float(region, 'addf', _float(region, 'mulf', x,
                       region.extract(cos, [pos, index])), product)]

    return builder.generic([], [builder.empty(value.shape, 'f32')], [identity_map(2)],
                           ['parallel', 'parallel'], body)[0]


def grouped_attention(context, config, q, k, v, caches, layer, position):
    from src.export.compilers.mlir.ops import attention
    from .math_kernels import SOFTMAX_EXP, softmax_exp_text
    if config.mode != 'naive' or config.kv != 'f32':
        raise ValueError('Imported GQA requires naive softmax and FP32 KV state')
    key = ('kernel', config.name)
    if key not in context.cache:
        if not getattr(context.graph, '_onnx_softmax_exp', False):
            context.graph.module.add_raw(softmax_exp_text())
            context.graph._onnx_softmax_exp = True
        context.graph.module.add_raw(attention.kernel_text(config, fastmath=False, fused=True,
            exp_function=SOFTMAX_EXP, sum_lanes=8, decode_group4=True))
        context.cache[key] = True
    return attention.attention(context, config, q, k, v, caches, layer, position)


def custom(context, node, inputs, *, config=None):
    """Dispatch a parsed ONNX custom node, rejecting unimplemented attributes."""
    if node['op'] != 'onnx.Custom':
        raise ValueError('Expected an imported ONNX custom operation')
    attrs = node['attributes']
    name = attrs['function_name']
    defaults = {
        'SimplifiedLayerNormalization': {'domain_name': '', 'axis': -1, 'stash_type': 1},
        'SkipSimplifiedLayerNormalization': {'domain_name': 'com.microsoft'},
        'RotaryEmbedding': {'domain_name': 'com.microsoft', 'interleaved': 0,
                            'num_heads': 0, 'rotary_embedding_dim': 0},
        'GroupQueryAttention': {'domain_name': 'com.microsoft', 'do_rotary': 0,
            'local_window_size': -1, 'rotary_interleaved': 0, 'softcap': 0},
    }
    if name not in defaults:
        raise NotImplementedError('No compiled lowering for imported custom operation ' + name)
    ignored = {'function_name', 'onnx_node_name'}
    parameters = {'epsilon'} if 'Normalization' in name else set()
    if name == 'GroupQueryAttention':
        parameters = {'num_heads', 'kv_num_heads', 'scale'}
    if set(attrs) - set(defaults[name]) - ignored - parameters:
        raise NotImplementedError('Unsupported imported attributes for ' + name)
    for key, value in defaults[name].items():
        if attrs.get(key, value) != value:
            raise NotImplementedError(f'Unsupported {name} attribute {key}={attrs[key]}')
    b = context.b
    if name == 'SimplifiedLayerNormalization':
        return (rms_norm(b, *inputs, epsilon=attrs.get('epsilon', 1e-5)),)
    if name == 'SkipSimplifiedLayerNormalization':
        return skip_norm(b, *inputs, epsilon=attrs.get('epsilon', 1e-5))
    if name == 'RotaryEmbedding':
        return (rotary(b, *inputs),)
    if config is None:
        raise ValueError('GQA requires a specialized cache/profile configuration')
    if (attrs.get('num_heads') != config.heads or attrs.get('kv_num_heads') != config.kv_heads
            or attrs.get('scale', 1.0 / config.head_dim ** 0.5) != config.scale):
        raise ValueError('GQA configuration differs from imported attributes')
    return (grouped_attention(context, config, *inputs),)


def sigmoid(builder, value):
    """Pinned logistic rational approximation with explicit fused evaluation.

    Coefficients are the Eigen/MLAS FP32 logistic approximation used by the
    pinned ORT reference. Explicit FMA fixes rounding across CPU targets.
    See ONNX Runtime v1.20.1 core/mlas/lib/logistic.cpp and
    x86_64/LogisticKernelFma3.S (MIT licensed).
    """
    def body(region):
        x = region.minf(region.maxf(region.args[0], region.const(-18.0, 'f32')),
                        region.const(18.0, 'f32'))
        square = _float(region, 'mulf', x, x)

        def polynomial(coefficients):
            result = region.const(coefficients[0], 'f32')
            for coefficient in coefficients[1:]:
                constant = region.const(coefficient, 'f32')
                result = region.op(f'math.fma {result}, {square}, {constant} : f32', 'f32')
            return result

        numerator = polynomial((4.37031012579801e-11, 1.15627324459942e-7,
                                6.08574864600143e-5, 8.51377133304701e-3, 2.48287947061529e-1))
        denominator = polynomial((6.10247389755681e-13, 5.76102136993427e-9,
            6.29106785017040e-6, 1.70198817374094e-3, 1.16817656904453e-1, 9.93151921023180e-1))
        ratio = _float(region, 'divf', _float(region, 'mulf', numerator, x), denominator)
        result = _float(region, 'addf', ratio, region.const(0.5, 'f32'))
        return region.maxf(result, region.const(0.0, 'f32'))

    return builder.elementwise([value], value.dtype, body)


def elementwise(builder, node, inputs):
    operation = node['op']
    if operation == 'onnx.Sigmoid' and len(inputs) == 1:
        return sigmoid(builder, inputs[0])
    raise NotImplementedError(f'No compiled lowering for imported operation {operation}')


def signed_contraction(context, node, activation, weight, name, zero_points=(0, 0)):
    import numpy as np

    if node['op'] != 'onnx.MatMulInteger':
        raise NotImplementedError(f'Not an imported integer contraction: {node["op"]}')
    if any(int(value) != 0 for value in zero_points):
        raise ValueError('Signed contraction requires zero zero-points')
    weight = np.asarray(weight)
    if activation.rank != 2 or activation.dtype != 'i8' or weight.ndim != 2 or weight.dtype != np.int8:
        raise ValueError('Signed contraction requires two rank-2 INT8 operands')
    rows, reduction = activation.shape
    if min(rows, reduction, *weight.shape) <= 0 or reduction != weight.shape[0]:
        raise ValueError('Signed contraction requires matching static dimensions')
    builder = context.b
    return builder.matmul(activation, context.weight(name, weight),
                          builder.fill(0, [rows, weight.shape[1]], 'i32'))


def native(ctx, name, inputs, outputs):
    """Keep native reads in tensor SSA until function-boundary bufferization.

    Early to_buffer casts hide later residual reads from One-Shot Bufferize:
    tensor.empty destinations can then reuse and overwrite those activations.
    External tensor arguments with explicit read access retain those use chains.
    """
    b = ctx.b
    buffers = []
    for value in inputs:
        if value.type.startswith('tensor<'):
            typ = memref_type([-1] * value.rank, value.dtype).replace('memref<','tensor<')
            value = b.op(f'tensor.cast {value} : {value.type} to {typ}',typ)
        elif value.type.startswith('memref<'):
            typ = memref_type([-1] * value.rank, value.dtype)
            value = b.op(f'memref.cast {value} : {value.type} to {typ}', typ)
        buffers.append(value)
    results = []
    for shape, dtype in outputs:
        typ = memref_type(shape, dtype)
        result = b.op(f'memref.alloc() : {typ}', typ)
        results.append(result)
        dynamic = memref_type([-1] * len(shape), dtype)
        buffers.append(b.op(f'memref.cast {result} : {typ} to {dynamic}', dynamic))
    declared = getattr(ctx.graph, '_native_symbols', set())
    if name not in declared:
        signature = ', '.join(f'%a{i}: {a.type}'+
            (' {bufferization.access = "read"}' if a.type.startswith('tensor<') else '')
            for i, a in enumerate(buffers))
        ctx.graph.module.add_raw(f'  func.func private @{name}({signature}) attributes {{llvm.emit_c_interface}}\n')
        declared.add(name)
        ctx.graph._native_symbols = declared
    b.call(name, buffers)
    return [b.to_tensor(result) for result in results]


def projection(ctx, activation, name, weight, scales):
    """Fuse original output columns, preserving UINT8 zero-point corrections."""
    b = ctx.b
    q, xs, zero = activation
    m, k = q.shape
    weight = np.asarray(weight, np.int8)
    n = weight.shape[1]
    centered = b.elementwise([q], 'i8', lambda r:
        r.trunci(r.subi(r.cast('arith.extui', r.args[0], 'i32'), r.const(128, 'i32'))))
    wq = ctx.weight(name + '.q', weight)
    sums = ctx.weight(name + '.sum', weight.sum(axis=0, dtype=np.int32))
    ws = ctx.weight(name + '.scale', np.asarray(scales, np.float32))
    acc = b.matmul(centered, wq, b.fill(0, [m, n], 'i32'))

    def finish(r):
        value, column_sum, az, act_scale, weight_scale = r.args[:5]
        az = r.cast('arith.extui', az, 'i32')
        correction = r.muli(r.subi(r.const(128, 'i32'), az), column_sum)
        integer = r.addi(value, correction)
        scale = r.binary('arith.mulf', act_scale, weight_scale, fast=False)
        return r.binary('arith.mulf', r.sitofp(integer), scale, fast=False)

    return b.elementwise([acc, sums, zero, xs, ws], 'f32', finish,
        maps=[identity_map(2), map_of(2, ['d1']), map_of(2, ['0']),
              map_of(2, ['0']), map_of(2, ['d1'])])


def multiply(b, x, y):
    return b.elementwise([x, y], 'f32', lambda r:
        r.binary('arith.mulf', r.args[0], r.args[1], fast=False))
