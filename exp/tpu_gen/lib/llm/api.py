from __future__ import annotations

import os

from tpugen_types import FlowError

from .base import LLMBackend, clean_vh

SYSTEM = (
    "You generate Verilog header files (.vh) for the TPU-Gen systolic array "
    "RTL. Reply with the header only: `define directives and `ifdef blocks, "
    "no prose, no markdown fences. Define the selected multiplier and adder "
    "as bare macros named for the chosen options, for example `define "
    "DRUM_APTPU and `define APPROX5. Never write generic `define MULTIPLIER "
    "or `define ADDER directives. Always define ROUN_WIDTH, NIBBLE_WIDTH, "
    "DW, WW, M, N, MULT_DW, ADDER_PARAM and VBL. For DRUM_APTPU, "
    "MITCHELL, ROBA, DRALM, ASM, or any ALM variant, also define "
    "SHARED_PRE_APPROX; the TPU-Gen top connects those branches to the "
    "shared pre-approximation unit."
)


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 2048):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise FlowError(
                "anthropic package not installed; pip install anthropic"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise FlowError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return clean_vh("".join(b.text for b in msg.content if b.type == "text"))


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 2048):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise FlowError("openai package not installed; pip install openai") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise FlowError("OPENAI_API_KEY is not set")
        self._client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return clean_vh(resp.choices[0].message.content or "")
