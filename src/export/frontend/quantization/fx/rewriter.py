"""The Rewriter: scheme-agnostic graph surgery from a plan (layer key -> QuantizedOp), in two passes.

    b. insert_qdq_op          x -> quant.quantize_per_tensor -> aten.linear(x_q, W_q, bias)
                                -> quant.dequantize(., s_a, s_w)
    a. mismatch_type_handler  x -> quant.quantize_per_tensor -> aten._int_mm(x_q, W_q)
                                -> quant.dequantize(., s_a, s_w) -> + bias

Pass b knows no op. It swaps the INT8 weight buffer in for the float weight
and, for a layer with an activation scale, inserts the quantize before the
op's activation input and the dequantize after the op. The op and its other
arguments (the bias too) are untouched, so the graph in between is never run
(an aten.linear on INT8 operands would overflow); it only has to be a valid FX
graph for pass a.

Pass a holds one rule per op: it replaces each op that now consumes INT8
operands with its integer form (symmetric INT8, zero point 0, for every
scheme; a scheme that needs another lowering adds its rule here):

    linear      the input reshaped to 2-D before the quantize, aten._int_mm, the
                float bias added after the dequantize, the result reshaped back
    conv2d      pad + im2col in (ky, kx, c) order before the quantize (the order
                the quantizer lays the conv weight out in), aten._int_mm, the
                float bias added after the dequantize, the result back to NCHW
    embedding   no activation quantization, so pass b only swapped in the INT8
                table; the lookup becomes rows of that table times their row
                scale

The layout steps are the ``_rows`` ... ``_lookup`` functions below, which the
re-export traces into ATen ops. The Rewriter decides no number: the tensors are
the plan's, and qmax (for quantize_per_tensor) is the scheme's.
"""

from __future__ import annotations

import operator

import torch
from torch import fx

from src.export.frontend.quantization.fx import custom_ops
from src.export.frontend.quantization.fx.utils.share import QuantizedOp, attr, layer_key, parse, select

# The custom ops' overloads, which a graph node calls (fx/custom_ops.py registers them).
QUANTIZE = getattr(torch.ops, custom_ops.NAMESPACE).quantize_per_tensor.default
DEQUANTIZE = getattr(torch.ops, custom_ops.NAMESPACE).dequantize.default


class Rewriter:
    """Rewrites every node of ``ops`` in each program in passes b then a; re-captures."""

    def __init__(self, ops, plan: dict[tuple, QuantizedOp], qmax: int) -> None:
        self.ops = tuple(ops)          # the op names whose nodes are rewritten
        self.plan = plan               # layer key -> QuantizedOp
        self.qmax = qmax               # the scheme's activation_qmax
        self.gm: fx.GraphModule | None = None   # the module being rewritten
        self.inserted: list[dict] = []          # per rewritten node, what pass b inserted, for pass a

    def run(self, programs: dict[str, torch.export.ExportedProgram]
            ) -> dict[str, torch.export.ExportedProgram]:
        out = {}
        for name, ep in programs.items():
            self.gm = ep.module()
            self.insert_qdq_op()
            self.mismatch_type_handler()
            self.finish()
            args, kwargs = ep.example_inputs
            with torch.no_grad():
                out[name] = torch.export.export(self.gm, args, kwargs, strict=False)
        return out

    # --- b ----------------------------------------------------------------------------

    def insert_qdq_op(self) -> None:
        """Pass b, knowing no op: the INT8 weight buffer in place of the float weight and, for
        a layer with an activation scale, the quantize before its input and the dequantize after."""
        graph, self.inserted = self.gm.graph, []
        for node in select(self.gm, self.ops):
            kind, args = parse(node)
            key = layer_key(self.gm, kind, args)
            if key not in self.plan:
                raise KeyError(f'no plan entry for {kind} weight {args["weight"].target}; '
                               'the plan must cover every node of the ops being replaced')
            weight = args['weight'].target
            entry = dict(node=node, kind=kind, args=args, shape=tuple(attr(self.gm, weight).shape))
            self.inserted.append(entry)
            with graph.inserting_before(node):
                entry['buffers'] = buffers = tuple(self._buffer(f'{weight}_{kind}_{i}', t)
                                                   for i, t in enumerate(self.plan[key].tensors()))
            node.replace_input_with(args['weight'], buffers[0])
            if self.plan[key].activation_scale is None:
                continue
            _, w_scale, x_scale = buffers
            with graph.inserting_before(node):
                entry['quantize'] = graph.call_function(QUANTIZE,
                                                        (args['x'], x_scale, self.qmax))
            node.replace_input_with(args['x'], entry['quantize'])
            users = list(node.users)
            with graph.inserting_after(node):
                entry['dequantize'] = graph.call_function(DEQUANTIZE, (node, x_scale, w_scale))
            for user in users:
                user.replace_input_with(node, entry['dequantize'])

    # --- a ----------------------------------------------------------------------------

    def mismatch_type_handler(self) -> None:
        """Pass a, one rule per op: each op on INT8 operands becomes its integer form, and a
        contraction's float bias moves after the dequantize (see the module docstring)."""
        graph = self.gm.graph
        for entry in self.inserted:
            node, kind, args, buffers = entry['node'], entry['kind'], entry['args'], entry['buffers']
            if kind == 'embedding':
                with graph.inserting_before(node):
                    new = graph.call_function(_lookup, (args['ids'], *buffers))
                node.replace_all_uses_with(new)
                graph.erase_node(node)
                continue
            quantize, last, x = entry['quantize'], entry['dequantize'], args['x']
            k, n = attr(self.gm, buffers[0].target).shape           # the INT8 weight's [K, N]
            if kind == 'linear':
                rows, back = (_rows, (x, k)), (_unrows, (x, n))
            else:
                layout = (entry['shape'][2:], args['stride'], args['padding'])
                rows, back = (_im2col, (x, *layout)), (_col2im, (x, *layout, n))
            with graph.inserting_before(quantize):
                quantize.update_arg(0, graph.call_function(rows[0], rows[1]))
            with graph.inserting_before(node):
                product = graph.call_function(torch._int_mm, (quantize, buffers[0]))
            node.replace_all_uses_with(product)
            graph.erase_node(node)
            users = list(last.users)
            if args['bias'] is not None:
                with graph.inserting_after(last):
                    last = graph.call_function(operator.add, (last, args['bias']))
            with graph.inserting_after(last):
                result = graph.call_function(back[0], (last, *back[1]))
            for user in users:
                user.replace_input_with(entry['dequantize'], result)

    # --- after both passes -------------------------------------------------------------

    def finish(self) -> None:
        """Drop dead nodes and the float weights no node reads, recompile.

        Without this, the re-export would keep the float weights."""
        gm = self.gm
        gm.graph.eliminate_dead_code()
        used = {n.target for n in gm.graph.nodes if n.op == 'get_attr'}
        # remove_duplicate=False lists every name of a tied weight (lm_head and the embedding).
        for name, _ in [*gm.named_parameters(remove_duplicate=False), *gm.named_buffers(remove_duplicate=False)]:
            if name not in used:
                owner, _, leaf = name.rpartition('.')
                delattr(gm.get_submodule(owner) if owner else gm, leaf)
        gm.recompile()

    def _buffer(self, hint: str, t: torch.Tensor) -> fx.Node:
        """A get_attr node for ``t``, registered as a buffer the first time it is used."""
        name = next((n for n, b in self.gm.named_buffers() if b is t), None)
        if name is None:
            name = hint.replace('.', '_')
            self.gm.register_buffer(name, t)
        return self.gm.graph.get_attr(name)


# --- the layout steps of pass a, traced into the graph -------------------------------

def _rows(x: torch.Tensor, k: int) -> torch.Tensor:
    """linear's input as the contraction's [M, K] rows: every token its own row."""
    return x.reshape(-1, k).to(torch.float32)


def _unrows(y: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    """The contraction's [M, N] back to linear's output shape: x's, with N last."""
    return y.reshape(*x.shape[:-1], n)


def _im2col(x: torch.Tensor, kernel, stride, padding) -> torch.Tensor:
    """conv2d's NCHW input -> [N * OH * OW, kh * kw * Cin] patches, K in (ky, kx, c) order.

    The order must be the one the quantizer lays the conv weight out in;
    otherwise the columns no longer line up with the weight's rows. Batch 1,
    no groups or dilation."""
    (kh, kw), (sh, sw), (ph, pw) = kernel, stride, padding
    x = torch.nn.functional.pad(x.to(torch.float32), (pw, pw, ph, ph))
    n, c, h, w = x.shape
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    patches = [x[:, :, ky:ky + sh * (oh - 1) + 1:sh, kx:kx + sw * (ow - 1) + 1:sw]
               for ky in range(kh) for kx in range(kw)]
    return torch.stack(patches, dim=1).permute(0, 3, 4, 1, 2).reshape(n * oh * ow, kh * kw * c)


def _col2im(y: torch.Tensor, x: torch.Tensor, kernel, stride, padding, cout: int) -> torch.Tensor:
    """The contraction's [N * OH * OW, Cout] back to NCHW."""
    (kh, kw), (sh, sw), (ph, pw) = kernel, stride, padding
    n, _, h, w = x.shape
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2)


def _lookup(ids: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """embedding on an INT8 table: the rows, rescaled by their row scale."""
    rows = qweight.index_select(0, ids.reshape(-1)).to(torch.float32)
    out = rows * scale.index_select(0, ids.reshape(-1))[:, None]
    return out.reshape(*ids.shape, qweight.shape[1])
