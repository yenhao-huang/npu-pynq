"""Token sampling: greedy, temperature, top-k, top-p, min-p, repetition penalty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SamplingParams:
    temperature: float = 0.0      # 0 selects greedy decoding
    top_k: int = 0                # 0 disables
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None


class Sampler:
    def __init__(self, params: SamplingParams) -> None:
        self.params = params
        self.rng = np.random.default_rng(params.seed)

    def __call__(self, logits: np.ndarray, history: list[int]) -> int:
        p = self.params
        logits = np.asarray(logits, np.float64).reshape(-1).copy()
        if p.repetition_penalty != 1.0 and history:
            seen = np.unique(np.asarray(history, np.int64))
            vals = logits[seen]
            logits[seen] = np.where(vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty)
        if p.temperature <= 0.0:
            return int(np.argmax(logits))
        logits /= p.temperature
        if p.top_k > 0 and p.top_k < logits.size:
            kth = np.partition(logits, -p.top_k)[-p.top_k]
            logits[logits < kth] = -np.inf
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if p.min_p > 0.0:
            probs[probs < p.min_p * probs.max()] = 0.0
        if p.top_p < 1.0:
            order = np.argsort(-probs)
            cumulative = np.cumsum(probs[order])
            cut = np.searchsorted(cumulative, p.top_p) + 1
            mask = np.zeros_like(probs, bool)
            mask[order[:cut]] = True
            probs[~mask] = 0.0
        probs /= probs.sum()
        return int(self.rng.choice(probs.size, p=probs))
