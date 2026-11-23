"""W8A8 as an FX transform: per-output-channel INT8 weights, static per-tensor INT8 activations.

The scheme's parameters are schemes.W8A8. Weights, per output channel, and
activations, one scale per layer input, are both symmetric with zero point 0:

    scale = max(amax, EPS) / 127    EPS avoids dividing by zero on an all-zero channel or input
    q     = round(value / scale)    half to even (torch.round), clamped to [-127, 127]

A weight's amax is its channel's; an activation's is the largest |input| the
layer saw while quantize.calibrate ran the float program on calibration
inputs (``calibrated_layers`` names those layers).

``rewrite`` replaces one aten.linear, aten.conv2d or aten.embedding node of an
unlifted GraphModule (weights as get_attr) with a call to ``linear``,
``conv2d`` or ``embedding`` below, after quantizing the weight and fixing the
activation scale once; both are stored as buffers. The re-export traces the
replacement functions; a contraction becomes

    xq  = quant.quantize_per_tensor(x, x_scale)    activations to INT8, constant [1] scale
    acc = aten._int_mm(xq, w)                      INT8 x INT8 -> INT32, constant [K, N] weight
    y   = quant.dequantize(acc, x_scale, w_scale)
    y   = y + bias

with the custom ops of fx/custom_ops.py. The embedding is a lookup in the INT8
table, in plain ATen ops, and has no activation scale.

Overflow: |acc| <= K * 127 * 127, which stays below 2**31 while K is under about
133,000, so an INT32 accumulator cannot overflow or saturate.
"""

from __future__ import annotations

import torch
from torch import fx

from src.export.frontend.quantization.fx import custom_ops
from src.export.frontend.quantization.schemes import W8A8

aten = torch.ops.aten

EPS = W8A8['eps']
QMAX = W8A8['activation_qmax']

# The scheme's op names (schemes.W8A8['ops']) -> the node each one rewrites.
TARGETS = {
    'linear': aten.linear.default,
    'conv2d': aten.conv2d.default,
    'embedding': aten.embedding.default,
}
KINDS = {target: op for op, target in TARGETS.items()}
# The ops whose input is quantized, and so calibrated.
CONTRACTIONS = ('linear', 'conv2d')


# --- weights and activation scales, fixed once ------------------------------

def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[out, in] float -> ([in, out] int8, [out] float32 scale), symmetric per output channel.

    Stored [K, N] so the contraction's right operand is the weight itself."""
    w = weight.detach().to(torch.float32).t().contiguous()           # [K, N]
    qmax = W8A8['weight_qmax']
    scale = torch.clamp(w.abs().amax(0), min=EPS) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_conv2d_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[Cout, Cin, kh, kw] float -> ([kh * kw * Cin, Cout] int8, [Cout] scale).

    K follows the (ky, kx, c) order that ``conv2d``'s im2col uses."""
    w = weight.detach().permute(0, 2, 3, 1).reshape(weight.shape[0], -1)   # [Cout, (ky, kx, c)]
    return quantize_weight(w)


def quantize_embedding_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[V, H] table -> ([V, H] int8, [V] scale): one scale per row."""
    q, s = quantize_weight(weight)      # per output channel of [V, H].T = per row
    return q.t().contiguous(), s


QUANTIZE_WEIGHT = {
    'linear': quantize_weight,
    'conv2d': quantize_conv2d_weight,
    'embedding': quantize_embedding_weight,
}


def activation_scale(amax: float) -> torch.Tensor:
    """The calibrated max |input| of a layer -> its [1] float32 activation scale."""
    return torch.clamp(torch.tensor([amax], dtype=torch.float32), min=EPS) / QMAX


# --- calibration ------------------------------------------------------------

def calibrated_layers(gm: fx.GraphModule, ops) -> dict[fx.Node, tuple]:
    """The contractions among ``ops``' nodes -> the key their statistic goes under.

    quantize.calibrate records each one's largest |input| (its first argument;
    for a convolution, before the padding, which adds zeros only). The key is
    the layer's weight's, the same one ``_quantized`` looks the statistic up by."""
    return {n: _key(gm, KINDS[n.target], n.args[1]) for n in select(gm, ops)
            if KINDS[n.target] in CONTRACTIONS}


# --- the graph rewrite ------------------------------------------------------

def select(gm: fx.GraphModule, ops) -> list[fx.Node]:
    """The nodes of ``gm`` that the op names ``ops`` cover, in graph order."""
    targets = {TARGETS[op] for op in ops}
    return [n for n in gm.graph.nodes if n.op == 'call_function' and n.target in targets]


def rewrite(gm: fx.GraphModule, node: fx.Node, cache: dict, stats: dict) -> None:
    """Replace ``node`` with the call that computes it in W8A8.

    ``cache`` maps each float weight to its quantized tensors and activation
    scale, and ``stats`` holds what quantize.calibrate recorded; the caller shares
    both across the programs of a model, so every program reads the same
    tensors for a weight it shares with another."""
    with gm.graph.inserting_before(node):
        if node.target == aten.linear.default:
            x, w, b = list(node.args) + [None] * (3 - len(node.args))   # aten.linear(x, weight, bias=None)
            qw, s, xs = _quantized(gm, 'linear', w, cache, stats)
            new = gm.graph.call_function(linear, (x, qw, s, xs, b))
        elif node.target == aten.conv2d.default:
            # aten.conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            defaults = [None, None, None, [1, 1], [0, 0], [1, 1], 1]
            x, w, b, stride, padding, dilation, groups = list(node.args) + defaults[len(node.args):]
            if groups != 1 or tuple(dilation) != (1, 1):
                raise ValueError('w8a8 supports ungrouped, undilated convolutions only')
            kernel = tuple(_attr(gm, w.target).shape[2:])
            qw, s, xs = _quantized(gm, 'conv2d', w, cache, stats)
            new = gm.graph.call_function(conv2d, (x, qw, s, xs, b, kernel, tuple(stride), tuple(padding)))
        elif node.target == aten.embedding.default:
            w, ids = node.args[:2]
            qw, s = _quantized(gm, 'embedding', w, cache, stats)
            new = gm.graph.call_function(embedding, (ids, qw, s))
        else:
            raise ValueError(f'w8a8 cannot rewrite {node.target}')
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)


def finish(gm: fx.GraphModule) -> None:
    """After every rewrite: drop dead nodes and the float weights no node reads, recompile.

    Without this, the re-export would keep the float weights."""
    gm.graph.eliminate_dead_code()
    used = {n.target for n in gm.graph.nodes if n.op == 'get_attr'}
    # remove_duplicate=False lists every name of a tied weight (lm_head and the embedding).
    for name, _ in [*gm.named_parameters(remove_duplicate=False), *gm.named_buffers(remove_duplicate=False)]:
        if name not in used:
            owner, _, leaf = name.rpartition('.')
            delattr(gm.get_submodule(owner) if owner else gm, leaf)
    gm.recompile()


def _attr(gm: fx.GraphModule, target: str) -> torch.Tensor:
    obj = gm
    for part in target.split('.'):
        obj = getattr(obj, part)
    return obj


def _key(gm: fx.GraphModule, kind: str, weight: fx.Node) -> tuple:
    """Where the float weight's data lives, before any transpose or permute.

    The same parameter reaches every program as the same storage, so this
    names a layer's weight across programs, unlike its module path."""
    if weight.op != 'get_attr':
        raise ValueError(f'{kind} weight {weight.name} is not a constant of the model')
    t = _attr(gm, weight.target)
    return (kind, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))


def _quantized(gm: fx.GraphModule, kind: str, weight: fx.Node, cache: dict, stats: dict) -> tuple:
    """get_attr nodes for the quantized ``weight`` (and, for a contraction, its
    activation scale), computing them only the first time."""
    key = _key(gm, kind, weight)
    if key not in cache:
        tensors = QUANTIZE_WEIGHT[kind](_attr(gm, weight.target))
        if kind in CONTRACTIONS:
            if key not in stats:
                raise ValueError(f'no calibration statistics for {kind} weight {weight.target}; '
                                 'w8a8 needs calibration inputs that reach every layer')
            tensors += (activation_scale(stats[key]),)
        cache[key] = tensors
    return tuple(_buffer(gm, f'{weight.target}_{kind}_{i}', q) for i, q in enumerate(cache[key]))


def _buffer(gm: fx.GraphModule, hint: str, t: torch.Tensor) -> fx.Node:
    """A get_attr node for ``t``, registered as a buffer of ``gm`` the first time it is used."""
    name = next((n for n, b in gm.named_buffers() if b is t), None)
    if name is None:
        name = hint.replace('.', '_')
        gm.register_buffer(name, t)
    return gm.graph.get_attr(name)


# --- the replacements, traced into the graph --------------------------------

def _contract(rows: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
              bias: torch.Tensor | None) -> torch.Tensor:
    """[M, K] float rows -> [M, N]: (int8(rows) @ W_q) * x_scale * w_scale + bias."""
    xq = custom_ops.quantize_per_tensor(rows, x_scale, QMAX)
    acc = torch._int_mm(xq, qweight)
    y = custom_ops.dequantize(acc, x_scale, scale)
    if bias is not None:
        y = y + bias
    return y


def linear(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
           bias: torch.Tensor | None) -> torch.Tensor:
    """aten.linear in W8A8."""
    shape = x.shape
    k, n = qweight.shape
    # The contraction is rank 2.
    y = _contract(x.reshape(-1, k).to(torch.float32), qweight, scale, x_scale, bias)
    return y.reshape(*shape[:-1], n)


def conv2d(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
           bias: torch.Tensor | None, kernel: tuple[int, int], stride: tuple[int, int],
           padding: tuple[int, int]) -> torch.Tensor:
    """A convolution as im2col and one W8A8 contraction.

    Rows are output pixels and K is kh * kw * Cin in (ky, kx, c) order.
    Batch 1, no groups or dilation.
    """
    (kh, kw), (sh, sw), (ph, pw) = kernel, stride, padding
    cout = qweight.shape[1]
    x = torch.nn.functional.pad(x.to(torch.float32), (pw, pw, ph, ph))
    n, c, h, w = x.shape
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    # K must follow the (ky, kx, c) order used in quantize_conv2d_weight.
    # Otherwise the im2col columns no longer line up with the weight's rows.
    patches = [x[:, :, ky:ky + sh * (oh - 1) + 1:sh, kx:kx + sw * (ow - 1) + 1:sw]
               for ky in range(kh) for kx in range(kw)]
    cols = torch.stack(patches, dim=1).permute(0, 3, 4, 1, 2).reshape(n * oh * ow, kh * kw * c)
    y = _contract(cols, qweight, scale, x_scale, bias)
    return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2)


def embedding(ids: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Rows of the INT8 table, rescaled. A table lookup, not a contraction."""
    rows = qweight.index_select(0, ids.reshape(-1)).to(torch.float32)
    out = rows * scale.index_select(0, ids.reshape(-1))[:, None]
    return out.reshape(*ids.shape, qweight.shape[1])
