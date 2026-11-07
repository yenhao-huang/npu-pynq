"""Text generation through an LLM package (manifest key ``llm``).

The KV cache is functional: ``decode``/``prefill`` read one cache set and
write the next, so the engine keeps two sets and swaps them per call.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from src.runtime.share.loader import ExportedPackage
from src.runtime.tasks.llm.sampling import Sampler, SamplingParams
from src.runtime.tasks.llm.tokenizer import Tokenizer


class ExportedLLM:
    def __init__(self, package: str | Path, overlay: Any = None) -> None:
        self.package = ExportedPackage(package, overlay)
        meta = self.package.manifest['llm']
        self.vocab, self.chunk, self.max_seq = meta['vocab'], meta['chunk'], meta['max_seq']
        shape = tuple(meta['cache']['shape'])
        count = 2 * meta['cache']['layers']
        self.cache = [np.zeros(shape, np.float32) for _ in range(count)]
        self.spare = [np.zeros(shape, np.float32) for _ in range(count)]
        self.logits = np.zeros((1, 1, self.vocab), np.float32)
        root = self.package.root
        self.tokenizer = Tokenizer(root / 'tokenizer.json') if (root / 'tokenizer.json').exists() else None
        self.eos = set(meta.get('eos', []))
        if self.tokenizer:
            for name in ('<|im_end|>', '<|endoftext|>'):
                tid = self.tokenizer.token_id(name)
                if tid >= 0:
                    self.eos.add(tid)
        self.pos = 0

    def close(self) -> None:
        self.package.close()

    def reset(self) -> None:
        for c in self.cache:
            c.fill(0)
        self.pos = 0

    def _step(self, entry: str, inputs: list[np.ndarray]) -> np.ndarray:
        self.package.call(entry, inputs + self.cache, [self.logits] + self.spare)
        self.cache, self.spare = self.spare, self.cache
        if not np.isfinite(self.logits).all():
            raise RuntimeError('the compiled model returned non-finite logits')
        return self.logits[0, -1]

    def prefill(self, ids: list[int]) -> np.ndarray:
        if self.pos + len(ids) > self.max_seq:
            raise ValueError(f'prompt exceeds the {self.max_seq}-token KV cache')
        logits = None
        for start in range(0, len(ids), self.chunk):
            part = ids[start:start + self.chunk]
            tokens = np.zeros((1, self.chunk), np.int64)
            tokens[0, :len(part)] = part
            logits = self._step('prefill', [tokens, np.array([self.pos], np.int64),
                                            np.array([len(part) - 1], np.int64)])
            self.pos += len(part)
        return logits

    def decode(self, token: int) -> np.ndarray:
        if self.pos >= self.max_seq:
            raise ValueError('KV cache is full')
        logits = self._step('decode', [np.array([[token]], np.int64), np.array([self.pos], np.int64)])
        self.pos += 1
        return logits

    def generate(self, prompt: str | list[int], max_new_tokens: int = 32, params: SamplingParams | None = None,
                 chat: bool = False, on_token: Callable[[int, str], None] | None = None):
        sampler = Sampler(params or SamplingParams())
        if isinstance(prompt, str):
            text = self.tokenizer.chat([{'role': 'user', 'content': prompt}]) if chat else prompt
            ids = self.tokenizer.encode(text)
        else:
            ids = list(prompt)
        self.reset()
        before = self.package.stats()
        t = time.perf_counter()
        logits = self.prefill(ids)
        prefill_s = time.perf_counter() - t
        history, out = list(ids), []
        t = time.perf_counter()
        for step in range(max_new_tokens):
            nxt = sampler(logits, history)
            if nxt in self.eos:
                break
            out.append(nxt)
            history.append(nxt)
            if on_token:
                on_token(nxt, self.tokenizer.decode([nxt]) if self.tokenizer else '')
            if step + 1 == max_new_tokens or self.pos >= self.max_seq:
                break
            logits = self.decode(nxt)
        decode_s = time.perf_counter() - t
        after = self.package.stats()
        stats = {'prompt_tokens': len(ids), 'generated_tokens': len(out),
                 'prefill_seconds': round(prefill_s, 3), 'decode_seconds': round(decode_s, 3),
                 'decode_tok_per_s': round(len(out) / decode_s, 3) if decode_s else 0.0,
                 'npu': {k: after[k] - before.get(k, 0) for k in after if k not in ('error', 'error_program')}}
        text = self.tokenizer.decode(out) if self.tokenizer else ''
        return out, text, stats
