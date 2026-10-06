"""Convolution and pooling for CNNs, on NHWC activations stored as [H*W, C].

A convolution is im2col (CPU, a gather) followed by a linear layer, so the
contraction becomes an NPU task: rows are output pixels, K is kh*kw*Cin.
"""

from __future__ import annotations

import numpy as np

from src.compiler.mlir_builder import Region, Value, identity_map, map_of
from src.compiler.ops.context import Ctx


def im2col(ctx: Ctx, x: Value, h: int, w: int, c: int, k: int, stride: int, pad: int) -> tuple[Value, int, int]:
    """x [h*w, c] -> [oh*ow, k*k*c], zero padded; column order (ky, kx, c)."""
    b = ctx.b
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    x3 = b.reshape(x, [h, w, c])

    def body(r: Region) -> Value:
        row, col = r.index(0), r.index(1)
        idx = lambda v: r.const(v, "index")  # noqa: E731

        def op(name, a, bb):
            return r.op(f"arith.{name} {a}, {bb} : index", "index")

        oy, ox = op("divui", row, idx(ow)), op("remui", row, idx(ow))
        ky = op("divui", col, idx(k * c))
        rest = op("remui", col, idx(k * c))
        kx, ch = op("divui", rest, idx(c)), op("remui", rest, idx(c))
        # Input coordinates in the padded frame; unsigned compare catches < 0.
        iy = op("subi", op("addi", op("muli", oy, idx(stride)), ky), idx(pad))
        ix = op("subi", op("addi", op("muli", ox, idx(stride)), kx), idx(pad))
        inside = r.op(f"arith.andi {r.cmpi('ult', iy, idx(h))}, {r.cmpi('ult', ix, idx(w))} : i1", "i1")
        cy = r.select(inside, iy, idx(0))
        cx = r.select(inside, ix, idx(0))
        v = r.extract(x3, [cy, cx, ch])
        return r.select(inside, v, r.const(0.0, "f32"))

    out = b.empty([oh * ow, k * k * c], "f32")
    return b.generic([], [out], [identity_map(2)], ["parallel", "parallel"], lambda r: [body(r)])[0], oh, ow


def maxpool(ctx: Ctx, x: Value, h: int, w: int, c: int, k: int, stride: int, pad: int) -> tuple[Value, int, int]:
    b = ctx.b
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    x3 = b.reshape(x, [h, w, c])

    def body(r: Region) -> Value:
        oy, ox, ch = r.index(0), r.index(1), r.index(2)
        idx = lambda v: r.const(v, "index")  # noqa: E731
        best = r.const(float("-inf"), "f32")
        for ky in range(k):
            for kx in range(k):
                iy = r.op(f"arith.subi {r.op(f'arith.addi {r.op(f'arith.muli {oy}, {idx(stride)} : index', 'index')}, {idx(ky)} : index', 'index')}, {idx(pad)} : index", "index")
                ix = r.op(f"arith.subi {r.op(f'arith.addi {r.op(f'arith.muli {ox}, {idx(stride)} : index', 'index')}, {idx(kx)} : index', 'index')}, {idx(pad)} : index", "index")
                inside = r.op(f"arith.andi {r.cmpi('ult', iy, idx(h))}, {r.cmpi('ult', ix, idx(w))} : i1", "i1")
                v = r.extract(x3, [r.select(inside, iy, idx(0)), r.select(inside, ix, idx(0)), ch])
                best = r.select(inside, r.maxf(best, v), best)
        return best

    out = b.empty([oh, ow, c], "f32")
    y = b.generic([], [out], [identity_map(3)], ["parallel"] * 3, lambda r: [body(r)])[0]
    return b.reshape(y, [oh * ow, c]), oh, ow


def global_avgpool(ctx: Ctx, x: Value) -> Value:
    """[n, c] -> [1, c] mean over rows."""
    b = ctx.b
    n, c = x.shape
    acc = b.fill(0.0, [1, c], "f32")
    s = b.generic([x], [acc], [map_of(2, ["d0", "d1"]), map_of(2, ["0", "d1"])],
                  ["reduction", "parallel"], lambda r: [r.addf(r.args[0], r.args[1])])[0]
    return b.elementwise([s], "f32", lambda r: r.mulf(r.args[0], r.const(1.0 / n, "f32")))


def ref_im2col(x: np.ndarray, h: int, w: int, c: int, k: int, stride: int, pad: int) -> np.ndarray:
    x3 = np.pad(x.reshape(h, w, c), ((pad, pad), (pad, pad), (0, 0)))
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    cols = np.empty((oh, ow, k, k, c), np.float32)
    for ky in range(k):
        for kx in range(k):
            cols[:, :, ky, kx] = x3[ky:ky + stride * oh:stride, kx:kx + stride * ow:stride]
    return cols.reshape(oh * ow, k * k * c)
