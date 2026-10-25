"""Lay out the weight arena: NPU-read weights as B tiles, everything else dense."""

from __future__ import annotations

import numpy as np

from src.vendors.npu_pynq.isa import isa, layout

WEIGHT_ALIGN = 64


def pack_weights(weights: dict[str, np.ndarray], meta: dict, report: dict,
                 limits: isa.Limits) -> tuple[bytes, dict, dict]:
    """Return (blob, {name: offset}, {name: layout}) with NPU weights tiled.

    ``weights`` and ``meta`` are a ModelGraph's fields of the same names."""
    npu_layout: dict[str, tuple[int, int, int]] = {}
    for task in report["npu_tasks"]:
        key = (task["k"], task["n"], task["tk"])
        prev = npu_layout.setdefault(task["weight"], key)
        if prev != key:
            raise ValueError(f"weight {task['weight']} used by NPU tasks with different tiling")
    for name, (k, n, tk) in meta.get("require_npu_layout", {}).items():
        if npu_layout.get(name) != (k, n, tk):
            raise ValueError(f"{name} is read in NPU layout (k={k}, n={n}, tk={tk}) but the partition "
                             f"stores it as {npu_layout.get(name, 'dense')}")
    packed = set(meta.get("packed_int4", []))
    chunks: list[bytes] = []
    offsets: dict[str, int] = {}
    layouts: dict[str, str] = {}
    cursor = 0
    for name, array in weights.items():
        if name in npu_layout:
            k, n, tk = npu_layout[name]
            if array.shape != (k, n) or array.dtype != np.int8:
                raise ValueError(f"NPU weight {name} must be int8 {k}x{n}, got {array.dtype} {array.shape}")
            plan = layout.plan_gemm(1, n, k, limits, tk)
            tiles = layout.pack_b(plan, array)
            if name in packed:
                if tiles.min() < -8 or tiles.max() > 7:
                    raise ValueError(f"{name} is marked INT4 but holds values outside [-8, 7]")
                u = tiles.view(np.uint8) & 0x0F
                data = (u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()
                layouts[name] = f"npu_b4(k={k},n={n},tk={tk})"
            else:
                data = tiles.tobytes()
                layouts[name] = f"npu_b(k={k},n={n},tk={tk})"
        else:
            data = np.ascontiguousarray(array).tobytes()
            layouts[name] = f"dense({array.dtype},{'x'.join(map(str, array.shape))})"
        pad = (-cursor) % WEIGHT_ALIGN
        chunks.append(b"\0" * pad)
        cursor += pad
        offsets[name] = cursor
        chunks.append(data)
        cursor += len(data)
    return b"".join(chunks), offsets, layouts
