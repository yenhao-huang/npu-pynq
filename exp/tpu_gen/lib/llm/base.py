from __future__ import annotations

import re
from abc import ABC, abstractmethod

from tpugen_types import FlowError

_FENCE = re.compile(r"^\s*```(?:\{?\.?vh\}?|verilog|systemverilog)?\s*$", re.M)


class LLMBackend(ABC):
    """Takes a formatted TPU-Gen prompt, returns Verilog header text."""

    name: str = "abstract"

    @abstractmethod
    def generate(self, prompt: str) -> str: ...


def clean_vh(text: str) -> str:
    """Strip markdown fences and the `{.vh}` marker the dataset carries."""
    text = text.replace("{.vh}", "")
    text = _FENCE.sub("", text)
    # Drop anything before the first `define / `ifdef / `ifndef.
    m = re.search(r"^\s*`(?:define|ifdef|ifndef|include)", text, re.M)
    if not m:
        raise FlowError(
            "LLM output contains no Verilog macro directive:\n"
            + text[:500]
        )
    return text[m.start():].strip() + "\n"
