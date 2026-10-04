"""Deterministic reference header.

Not an LLM substitute: used to smoke-test the OpenROAD path and as the
known-good baseline a generated header is compared against.
"""

from __future__ import annotations

from prompt_formatter import DesignSpec

_TEMPLATE = """\
`define {multiplier}
`define {adder}
`define ROUN_WIDTH 1
`define NIBBLE_WIDTH 4
`define DW {dw}
`define WW {ww}
`define M {m}
`define N {n}
`define MULT_DW {mult_dw}
`define ADDER_PARAM {adder_param}
`define VBL 16

`ifdef NORMAL_TPU
    `define ACCURATE_ACCUMULATE
`endif

`ifdef MITCHELL
    `define SHARED_PRE_APPROX
`elsif ALM_MAA3
    `define SHARED_PRE_APPROX
`elsif ALM_SOA
    `define SHARED_PRE_APPROX
`elsif ALM_LOA
    `define SHARED_PRE_APPROX
`elsif ROBA
    `define SHARED_PRE_APPROX
`elsif DRUM_APTPU
    `define SHARED_PRE_APPROX
`elsif ALM
    `define SHARED_PRE_APPROX
`elsif DRALM
    `define SHARED_PRE_APPROX
`elsif ASM
    `define SHARED_PRE_APPROX
`endif

`ifdef ALM_MAA3
    `define ALM
`elsif ALM_SOA
    `define ALM
`elsif ALM_LOA
    `define ALM
`endif
"""


def render_reference_vh(spec: DesignSpec, *, adder_param: int | None = None) -> str:
    spec.validate()
    return _TEMPLATE.format(
        multiplier=spec.multiplier,
        adder=spec.adder,
        dw=spec.dw,
        ww=spec.ww,
        m=spec.m,
        n=spec.n,
        mult_dw=spec.mult_dw,
        adder_param=adder_param if adder_param is not None else max(spec.dw, spec.ww),
    )
