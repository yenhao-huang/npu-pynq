"""Run an exported LLM package: prefill, decode, KV cache and sampling.

The model's kernels are the compiled package (CPU code from LLVM, NPU tasks
from the ISA encoder); this module only schedules them: it owns the KV cache
(and any recurrent state) as host buffers, feeds the prompt through the
fixed-size prefill entry point chunk by chunk, then decodes one token per
call and samples from the logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.inference.graph import np_dtype
from src.inference.sampling import Sampler, SamplingParams
from src.inference.tokenizer import Tokenizer
from src.runtime.compiled import CompiledModel, platform_backend


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    npu: dict = field(default_factory=dict)

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tps(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds else 0.0

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "generated_tokens": self.generated_tokens,
                "prefill_seconds": round(self.prefill_seconds, 4), "decode_seconds": round(self.decode_seconds, 4),
                "prefill_tok_per_s": round(self.prefill_tps, 3), "decode_tok_per_s": round(self.decode_tps, 3),
                "npu": self.npu}


class LLMEngine:
    def __init__(self, package: str | Path, backend: str | None = None, overlay=None,
                 stream_weights: bool | None = None) -> None:
        self.package = Path(package)
        self.backend = backend or platform_backend()
        self.model = CompiledModel(self.package, self.backend, overlay=overlay, stream_weights=stream_weights)
        meta = self.model.manifest["meta"]
        self.meta = meta
        self.vocab = meta["vocab"]
        self.chunk = meta["chunk"]
        self.max_seq = meta["options"]["max_seq"]
        self.state = [np.zeros(s["shape"], np_dtype(s["dtype"])) for s in meta["state"]]
        self.logits = np.zeros((1, self.vocab), np.float32)
        self.tokens_buf = np.zeros(self.chunk, np.int32)
        tok = self.package / "tokenizer.json"
        self.tokenizer = Tokenizer(tok) if tok.exists() else None
        eos = meta.get("eos")
        self.eos = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        if self.tokenizer:
            for name in ("<|im_end|>", "<|endoftext|>"):
                tid = self.tokenizer.token_id(name)
                if tid >= 0:
                    self.eos.add(tid)
        self.pos = 0

    def reset(self) -> None:
        for s in self.state:
            s.fill(0)
        self.pos = 0

    # ------------------------------------------------------------ kernels
    def prefill(self, ids: list[int]) -> np.ndarray:
        """Feed ``ids`` at the current position; return the last token's logits."""
        if self.pos + len(ids) > self.max_seq:
            raise ValueError(f"prompt exceeds the {self.max_seq}-token KV cache")
        # Feed fixed-size chunks; only real tokens advance the cache position.
        for start in range(0, len(ids), self.chunk):
            part = ids[start:start + self.chunk]
            self.tokens_buf[:] = 0
            self.tokens_buf[:len(part)] = part
            self.model.call("prefill", "weights", *self.state, self.tokens_buf, self.pos, len(part) - 1,
                            self.logits)
            self.pos += len(part)
        return self.logits[0]

    def decode(self, token: int) -> np.ndarray:
        if self.pos >= self.max_seq:
            raise ValueError("KV cache is full")
        # One compiled call updates the cache and returns the next logits.
        self.model.call("decode", "weights", *self.state, int(token), self.pos, self.logits)
        self.pos += 1
        return self.logits[0]

    # --------------------------------------------------------- generation
    def generate(self, prompt: str | list[int], max_new_tokens: int = 32,
                 params: SamplingParams | None = None, chat: bool = False,
                 on_token: Callable[[int, str], None] | None = None,
                 use_chunked_prefill: bool = True) -> tuple[list[int], str, GenerationStats]:
        params = params or SamplingParams()
        sampler = Sampler(params)
        # Apply the chat template if requested, then tokenize the prompt.
        if isinstance(prompt, str):
            text = self.tokenizer.chat([{"role": "user", "content": prompt}]) if chat else prompt
            ids = self.tokenizer.encode(text)
        else:
            ids = list(prompt)

        # Clear the KV cache before processing this prompt.
        self.reset()
        stats = GenerationStats(prompt_tokens=len(ids))
        before = self.model.stats()
        t = time.perf_counter()
        # Process the prompt in chunks, or token by token for comparison.
        if use_chunked_prefill:
            logits = self.prefill(ids)
        else:
            for tid in ids:
                logits = self.decode(tid)
        stats.prefill_seconds = time.perf_counter() - t
        history = list(ids)
        out: list[int] = []
        t = time.perf_counter()
        # Sample from the current logits, then decode that token for the next step.
        for _ in range(max_new_tokens):
            nxt = sampler(logits, history)
            if nxt in self.eos:
                break
            out.append(nxt)
            history.append(nxt)
            if on_token:
                on_token(nxt, self.tokenizer.decode([nxt]) if self.tokenizer else "")
            if self.pos >= self.max_seq:
                break
            logits = self.decode(nxt)
        stats.decode_seconds = time.perf_counter() - t
        stats.generated_tokens = len(out)
        after = self.model.stats()
        stats.npu = {k: after[k] - before[k] for k in ("calls", "jobs", "cycles", "macs")}
        # Convert generated IDs to text and return them with timing and NPU stats.
        text = self.tokenizer.decode(out) if self.tokenizer else ""
        return out, text, stats

    def score(self, ids: Iterable[int]) -> list[np.ndarray]:
        """Logits after each token (decode path), for evaluation."""
        self.reset()
        return [self.decode(t).copy() for t in ids]
