"""torch.export for a decoder-only LM: one ExportedProgram per entry point.

    decode (tokens [1, 1], pos [1], *cache)          -> (logits [1, 1, V], *cache')
    prefill(tokens [1, T], pos [1], last [1], *cache) -> (logits [1, 1, V], *cache')

The KV cache is explicit: ``cache`` is one key and one value tensor per layer,
[1, kv_heads, max_seq, head_dim], and the function returns the updated ones
instead of writing in place, so the graph stays functional and lowers to
plain linalg. The runtime passes two cache sets and swaps them per call.
``prefill`` writes positions pos .. pos + T - 1 (padding past ``last`` writes
positions later tokens overwrite before reading) and returns the logits of
token ``last``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CacheSpec:
    layers: int
    kv_heads: int
    max_seq: int
    head_dim: int

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, self.kv_heads, self.max_seq, self.head_dim)

    def names(self) -> list[str]:
        return [f'{kind}{i}' for i in range(self.layers) for kind in ('k', 'v')]

    def zeros(self) -> list[torch.Tensor]:
        return [torch.zeros(self.shape) for _ in range(2 * self.layers)]


def cache_spec(model: nn.Module, max_seq: int) -> CacheSpec:
    c = model.config
    head_dim = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
    return CacheSpec(c.num_hidden_layers, c.num_key_value_heads, max_seq, head_dim)


def functional_cache(spec: CacheSpec, kv, pos):
    from transformers.cache_utils import Cache

    class FunctionalCache(Cache):
        """update() returns new tensors: positions pos .. pos + T - 1 take this step's rows."""

        def __init__(self):
            self.kv, self.layers = list(kv), []

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            t = key_states.shape[2]
            rel = torch.arange(spec.max_seq) - pos[0]
            take = ((rel >= 0) & (rel < t)).view(1, 1, spec.max_seq, 1)
            idx = rel.clamp(0, t - 1)
            for j, new in ((2 * layer_idx, key_states), (2 * layer_idx + 1, value_states)):
                self.kv[j] = torch.where(take, new.index_select(2, idx), self.kv[j])
            return self.kv[2 * layer_idx], self.kv[2 * layer_idx + 1]

        def get_seq_length(self, layer_idx=0):
            return 0

        def get_mask_sizes(self, cache_position, layer_idx=0):
            return spec.max_seq, 0

        @property
        def is_compileable(self):
            return True

        @property
        def is_sliding(self):
            return [False] * spec.layers

        def __len__(self):
            return spec.layers

    return FunctionalCache()


class Step(nn.Module):
    """One call of the LM over T tokens at ``pos`` with an explicit cache."""

    def __init__(self, model: nn.Module, spec: CacheSpec, prefill: bool) -> None:
        super().__init__()
        self.model, self.spec, self.prefill = model, spec, prefill

    def forward(self, tokens, pos, *rest):
        last, kv = (rest[0], rest[1:]) if self.prefill else (None, rest)
        t = tokens.shape[1]
        cache = functional_cache(self.spec, kv, pos)
        # Additive causal mask over the whole cache: query i sees s <= pos + i.
        s = torch.arange(self.spec.max_seq).view(1, -1)
        q = (pos[0] + torch.arange(t)).view(-1, 1)
        mask = torch.where(s <= q, 0.0, float('-inf')).view(1, 1, t, self.spec.max_seq)
        positions = pos[0] + torch.arange(t)
        out = self.model(input_ids=tokens, position_ids=positions.view(1, -1), cache_position=positions,
                         past_key_values=cache, attention_mask=mask, use_cache=True,
                         logits_to_keep=last if last is not None else 1)
        return (out.logits, *cache.kv)


def example_inputs(spec: CacheSpec, tokens: int, prefill: bool) -> tuple:
    head = (torch.zeros(1, tokens, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    if prefill:
        head += (torch.full((1,), tokens - 1, dtype=torch.long),)
    return head + tuple(spec.zeros())


def export_llm(model: nn.Module, max_seq: int, chunk: int) -> tuple[dict[str, torch.export.ExportedProgram], CacheSpec]:
    """{'decode': ep, 'prefill': ep} after the default ATen decompositions."""
    spec = cache_spec(model, max_seq)
    programs = {}
    for name, tokens, prefill in (('decode', 1, False), ('prefill', chunk, True)):
        step = Step(model, spec, prefill)
        with torch.no_grad():
            ep = torch.export.export(step, example_inputs(spec, tokens, prefill), strict=False)
        programs[name] = ep.run_decompositions()
    return programs, spec


class Classify(nn.Module):
    """forward(image [H, W, C] normalized NHWC) -> logits [1, classes]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image.permute(2, 0, 1).unsqueeze(0))


def export_classifier(model: nn.Module, shape: tuple[int, int, int]) -> dict[str, torch.export.ExportedProgram]:
    """{'forward': ep} for an image classifier taking one NHWC image."""
    with torch.no_grad():
        ep = torch.export.export(Classify(model), (torch.zeros(shape),), strict=False)
    return {'forward': ep.run_decompositions()}
