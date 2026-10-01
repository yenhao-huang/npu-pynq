"""VCD debug backend.

The fallback the architecture notes call for, and the direct path for traces
produced by the `icarus` sim backend. VCD is 10-100x larger than FST and has
no random access, but it needs no external converter, so it is the backend
that always works.
"""

from __future__ import annotations

from pathlib import Path

from ...registry import backend
from .engine import OPS, Wave


class _VcdBase:
    """Dispatch the four ops onto the shared engine."""

    def _wave(self, params, ctx) -> Wave:
        return Wave(self._prepare(Path(ctx.resolve(params.wave))))

    def _prepare(self, path: Path) -> Path:
        return path

    def signals(self, params, ctx):
        return OPS["signals"](self._wave(params, ctx), params)

    def value_at(self, params, ctx):
        return OPS["value_at"](self._wave(params, ctx), params)

    def value_range(self, params, ctx):
        return OPS["value_range"](self._wave(params, ctx), params)

    def first_mismatch(self, params, ctx):
        return OPS["first_mismatch"](self._wave(params, ctx), params)


@backend("debug", "vcd")
class VcdDebug(_VcdBase):
    pass
