"""Export options, split by the stage that consumes them.

``QuantRecipe`` decides the numbers (quantize), ``ShapeOptions`` the graph's
shapes and kernels (to_mlir). ``LLMOptions`` keeps the original combined
constructor for the CLI and existing callers, and its ``describe()`` is what
packages record as ``meta['options']``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.vendors.npu_pynq.isa import isa


@dataclass
class QuantRecipe:
    bits: int = 8                # linear weights: 8 (per channel), 4 (group-wise) or 32 (float reference)
    group: int = 128             # INT4 group size along K; resolved per model by quantize
    embed_bits: int = 8          # tied embedding / lm_head
    awq: bool = False            # activation-aware scaling before quantization
    awq_samples: list[str] = field(default_factory=list)


@dataclass
class ShapeOptions:
    chunk: int = 16              # prefill tokens per call
    max_seq: int = 512           # KV cache positions
    attention: str = "online"    # "online" or "naive"
    kv: str = "f32"              # "f32" or "int8" KV cache
    prefill_lengths: tuple[int, ...] = (36, 40, 42)   # fixed prefill lengths (ONNX source)
    limits: isa.Limits = field(default_factory=isa.Limits)


@dataclass
class LLMOptions:
    chunk: int = 16              # prefill tokens per call
    max_seq: int = 512           # KV cache positions
    bits: int = 8                # linear weights: 8 (per channel) or 4 (group-wise)
    group: int = 128             # INT4 group size along K
    embed_bits: int = 8          # tied embedding / lm_head
    kv: str = "f32"              # "f32" or "int8" KV cache
    attention: str = "online"    # "online" or "naive"
    awq: bool = False            # activation-aware scaling before quantization
    awq_samples: list[str] = field(default_factory=list)
    layers: int | None = None    # debug: keep only the first N decoder layers
    limits: isa.Limits = field(default_factory=isa.Limits)

    def describe(self) -> dict:
        d = asdict(self)
        d.pop("awq_samples")
        return d

    def split(self, prefill_lengths: tuple[int, ...] = (36, 40, 42)) -> tuple[QuantRecipe, ShapeOptions, int | None]:
        recipe = QuantRecipe(self.bits, self.group, self.embed_bits, self.awq, list(self.awq_samples))
        shape = ShapeOptions(self.chunk, self.max_seq, self.attention, self.kv, tuple(prefill_lengths), self.limits)
        return recipe, shape, self.layers

    @classmethod
    def join(cls, recipe: QuantRecipe, shape: ShapeOptions, layers: int | None) -> "LLMOptions":
        return cls(shape.chunk, shape.max_seq, recipe.bits, recipe.group, recipe.embed_bits, shape.kv,
                   shape.attention, recipe.awq, list(recipe.awq_samples), layers, shape.limits)
