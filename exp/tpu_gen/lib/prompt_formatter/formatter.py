from __future__ import annotations

import re
from dataclasses import dataclass

from tpugen_types import PPA, PPATarget

# Selector macros the RTL switches on. Kept in sync with
# src_code/rtl/options_definitions.vh and RAG.ipynb cell 7.
AP_MULTIPLIERS = [
    "DRUM_APTPU", "BAM", "UDM", "EIM", "ALM_MAA3", "ALM", "ALM_SOA",
    "ALM_LOA", "MITCHELL", "ROBA", "DRALM", "ASM", "HIGH_REG",
]
AP_ADDERS = [
    "HERLOA", "OLOCA4", "SETA", "MHERLOA", "MHEAA", "LZTA", "LOAWA",
    "LOA", "HOERAA", "HOAANED", "HEAA", "APPROX5", "SA_ADDER", "LDCA",
]

_DESCRIPTION = (
    "This is the Verilog Header file that contains the design of {m}x{n} "
    "systolic array implementation. It contains the following {mul} "
    "approximate multiplier, with the following {add} approximate adder. "
    "It has the support of the following {dw} dataflow as input, and "
    "supports input weights {ww} bits of integer. With a low precision "
    "multiplier coefficient of {mult_dw} employed in this device."
)

_HEADER = "Generate Verilog header file based on the following description and metrics:"


@dataclass
class DesignSpec:
    """The knobs a TPU-Gen description encodes."""

    m: int = 4
    n: int = 4
    multiplier: str = "DRUM_APTPU"
    adder: str = "APPROX5"
    dw: int = 8
    ww: int = 8
    mult_dw: int = 4

    def validate(self) -> None:
        from tpugen_types import FlowError

        if self.multiplier not in AP_MULTIPLIERS:
            raise FlowError(
                f"unknown multiplier {self.multiplier!r}; "
                f"expected one of {AP_MULTIPLIERS}"
            )
        if self.adder not in AP_ADDERS:
            raise FlowError(
                f"unknown adder {self.adder!r}; expected one of {AP_ADDERS}"
            )
        for name in ("m", "n", "dw", "ww", "mult_dw"):
            if getattr(self, name) <= 0:
                raise FlowError(f"{name} must be positive, got {getattr(self, name)}")
        if self.mult_dw > min(self.dw, self.ww):
            raise FlowError(
                f"MULT_DW ({self.mult_dw}) cannot exceed min(DW, WW) "
                f"({min(self.dw, self.ww)})"
            )

    def describe(self) -> str:
        return _DESCRIPTION.format(
            m=self.m, n=self.n, mul=self.multiplier, add=self.adder,
            dw=self.dw, ww=self.ww, mult_dw=self.mult_dw,
        )


def parse_user_prompt(text: str, default: DesignSpec | None = None) -> DesignSpec:
    """Pull design knobs out of a free-form request.

    Anything not mentioned keeps the default, so "a 4x4 INT8 TPU with a BAM
    multiplier" is a complete request.
    """
    spec = default or DesignSpec()
    upper = text.upper()

    if (m := re.search(r"(\d+)\s*[xX*]\s*(\d+)", text)):
        spec.m, spec.n = int(m.group(1)), int(m.group(2))

    # Longest name first so ALM_LOA is not swallowed by ALM.
    for name in sorted(AP_MULTIPLIERS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", upper):
            spec.multiplier = name
            break
    for name in sorted(AP_ADDERS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", upper):
            spec.adder = name
            break

    if (m := re.search(r"INT\s*(\d+)", upper)):
        spec.dw = spec.ww = int(m.group(1))
    if (m := re.search(r"(?:IFMAP|INPUT|DATA)\s*(?:WIDTH|BITS?)?\s*(?:OF\s*)?(\d+)\s*BITS?", upper)):
        spec.dw = int(m.group(1))
    if (m := re.search(r"WEIGHTS?\s*(?:OF\s*)?(\d+)\s*BITS?", upper)):
        spec.ww = int(m.group(1))
    if (m := re.search(r"COEFFICIENT\s*(?:OF\s*)?(\d+)", upper)):
        spec.mult_dw = int(m.group(1))

    spec.mult_dw = min(spec.mult_dw, spec.dw, spec.ww)
    spec.validate()
    return spec


def format_prompt(
    spec: DesignSpec,
    target: PPATarget,
    *,
    previous_ppa: PPA | None = None,
    errors: list[str] | None = None,
) -> str:
    """Render the training-format prompt, plus feedback on later iterations."""
    spec.validate()
    parts = [
        _HEADER,
        f"Description: {spec.describe()}",
        f"Metrics: {target.as_prompt_metrics()}",
    ]

    if previous_ppa is not None:
        d = previous_ppa.distance(target)
        parts += [
            "",
            "The previous attempt was synthesized and produced:",
            f"Area: {previous_ppa.area:g} (off target by {d['area']:.1%}), "
            f"WNS: {previous_ppa.wns:g} (off by {d['wns']:.1%}), "
            f"Total Power: {previous_ppa.total_power:.2e} (off by {d['total_power']:.1%}).",
            "Adjust the header so the metrics move toward the target.",
        ]

    if errors:
        parts += ["", "The previous attempt failed with:"]
        parts += [f"- {e}" for e in errors[:10]]
        parts.append("Produce a header that avoids these errors.")

    return "\n".join(parts)
