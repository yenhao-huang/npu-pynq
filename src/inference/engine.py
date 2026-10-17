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
from src.inference.graph import ONNXGraph, np_dtype
from src.inference.sampling import Sampler, SamplingParams
from src.inference.tokenizer import Tokenizer
from src.runtime.compiled import CompiledModel, platform_backend
from types import SimpleNamespace
from src.inference.weights import onnx_asset_dir


@dataclass
class GenerationStats(dict):
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    npu: dict = field(default_factory=dict)
    per_token_seconds: list[float] = field(default_factory=list)
    compiled_pipeline: str | None = None

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tps(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds else 0.0

    def as_dict(self) -> dict:
        result = {"prompt_tokens": self.prompt_tokens, "generated_tokens": self.generated_tokens,
                "prefill_seconds": round(self.prefill_seconds, 4), "decode_seconds": round(self.decode_seconds, 4),
                "prefill_tok_per_s": round(self.prefill_tps, 3), "decode_tok_per_s": round(self.decode_tps, 3),
                "npu": self.npu}
        if self.compiled_pipeline:
            result.update(per_token_seconds=self.per_token_seconds,
                          compiled_pipeline=self.compiled_pipeline)
        return result


class LLMEngine:
    def __init__(self, package: str | Path, backend: str | None = None, overlay=None,
                 stream_weights: bool | None = None) -> None:
        self.package = Path(package)
        self.backend = backend or platform_backend()
        self.model = CompiledModel(self.package, self.backend, overlay=overlay, stream_weights=stream_weights)
        meta = self.model.manifest["meta"]
        self.meta = meta
        self.vocab = meta["vocab"]
        self.prefill_lengths = meta.get('prefill_lengths', ())
        self.chunk = meta.get('chunk', max(self.prefill_lengths, default=1))
        self.max_seq = meta["options"]["max_seq"]
        self.state = [np.zeros(s["shape"], np_dtype(s["dtype"])) for s in meta["state"]]
        self.logits = np.zeros((1, self.vocab), np.float32)
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
        self.graph = SimpleNamespace(integer_gemm=_PhysicalStats(self.model) if self.backend == 'pynq' else None)
        self.reset()

    def reset(self) -> None:
        for s in self.state:
            s.fill(0)
        self.pos = 0
        self.model.reset()

    def _call_tokens(self, entry: str, ids: list[int]) -> None:
        """Bind the graph's declared ABI to the same cache and token buffers."""
        values = {'weights':'weights', 'tokens':np.asarray(ids, np.int32),
                  'token':ids[0], 'pos':self.pos, 'last':len(ids)-1, 'logits':self.logits}
        values.update({s['name']:value for s, value in zip(self.meta['state'], self.state)})
        self._call(entry, values)
        if not np.isfinite(self.logits).all():
            raise RuntimeError('Compiled model returned nonfinite logits')
        self.pos += len(ids)

    def _call(self, entry, values):
        args = []
        for name in self.model.manifest['entry_points'][entry]:
            # The original manifest used one placeholder for all state buffers.
            args.extend(self.state if name == 'state' else [values[name]])
        self.model.call(entry, *args)

    # ------------------------------------------------------------ kernels
    def prefill(self, ids: list[int]) -> np.ndarray:
        """Feed ``ids`` at the current position; return the last token's logits."""
        if self.pos + len(ids) > self.max_seq:
            raise ValueError(f"prompt exceeds the {self.max_seq}-token KV cache")
        if not ids:
            raise ValueError('Empty input')
        # Dynamic activation quantization requires the entire unpadded prompt.
        if self.prefill_lengths:
            if self.pos or len(ids) not in self.prefill_lengths:
                raise ValueError(f'Exact prefill length {len(ids)} is not compiled; supported: {self.prefill_lengths}')
            self._call_tokens(f'prefill_{len(ids)}', ids)
            return self.logits
        # Feed fixed-size chunks; only real tokens advance the cache position.
        for start in range(0, len(ids), self.chunk):
            part = ids[start:start + self.chunk]
            padded = np.zeros(self.chunk, np.int32)
            padded[:len(part)] = part
            values = {'weights':'weights', 'tokens':padded, 'pos':self.pos,
                      'last':len(part)-1, 'logits':self.logits}
            values.update({s['name']:value for s,value in zip(self.meta['state'],self.state)})
            self._call('prefill', values)
            self.pos += len(part)
        return self.logits[0]

    def decode(self, token: int) -> np.ndarray:
        if self.pos >= self.max_seq:
            raise ValueError("KV cache is full")
        # One compiled call updates the cache and returns the next logits.
        self._call_tokens('decode', [int(token)])
        return self.logits if self.prefill_lengths else self.logits[0]

    def forward(self, ids):
        """Compatibility entry point for full-prompt reference traces."""
        if not ids:
            raise ValueError('Empty input')
        return self.decode(ids[0]) if len(ids) == 1 else self.prefill(ids)

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
        for step in range(max_new_tokens):
            nxt = sampler(logits, history)
            if nxt in self.eos:
                break
            out.append(nxt)
            history.append(nxt)
            if on_token:
                on_token(nxt, self.tokenizer.decode([nxt]) if self.tokenizer else "")
            if self.pos >= self.max_seq or step + 1 == max_new_tokens:
                break
            token_start = time.perf_counter()
            logits = self.decode(nxt)
            stats.per_token_seconds.append(time.perf_counter() - token_start)
        stats.decode_seconds = time.perf_counter() - t
        stats.generated_tokens = len(out)
        after = self.model.stats()
        stats.npu = {k: after[k] - before[k] for k in ("calls", "jobs", "cycles", "macs")}
        stats.compiled_pipeline = self.meta.get('compiled_pipeline')
        stats.update(stats.as_dict())
        # Convert generated IDs to text and return them with timing and NPU stats.
        text = self.tokenizer.decode(out) if self.tokenizer else ""
        return out, text, stats

    def score(self, ids: Iterable[int]) -> list[np.ndarray]:
        """Logits after each token (decode path), for evaluation."""
        self.reset()
        return [self.decode(t).copy() for t in ids]


class ONNXReferenceEngine:
    def __init__(self, model_dir, integer_gemm=None, float_ops=None):
        model_dir = Path(model_dir)
        # Accept the previous experiment bundle layout for existing notebooks.
        if not (onnx_asset_dir(model_dir, "onnx-graph")).is_dir():
            model_dir = model_dir / "packages"
        self.graph = ONNXGraph(onnx_asset_dir(model_dir, "onnx-graph"), integer_gemm, float_ops)
        self.tokenizer = Tokenizer(onnx_asset_dir(model_dir, "onnx"))
        self.max_seq = 512
        self.reset()

    def reset(self):
        self.past = {f"past_key_values.{i}.{kind}": np.empty((1,3,0,64), np.float32)
                     for i in range(30) for kind in ("key", "value")}
        self.logits = None

    def forward(self, ids):
        length = next(iter(self.past.values())).shape[2]
        if length + len(ids) > self.max_seq:
            raise ValueError("Context exceeds configured board memory limit")
        result = self.graph.run({"input_ids": np.asarray([ids], np.int64),
                                 "attention_mask": np.ones((1,length+len(ids)), np.int64),
                                 "position_ids": np.arange(length,length+len(ids),dtype=np.int64)[None], **self.past})
        self.past = {k.replace("present.","past_key_values."): v for k,v in result.items() if k != "logits"}
        self.logits = result["logits"][:, -1]
        return self.logits

    def prefill(self, ids):
        return self.forward(ids)

    def decode(self, token):
        return self.forward([token])

    def generate(self, prompt, max_new_tokens=24, chat=True, on_token=None):
        self.reset()
        ids = self.tokenizer.encode(self.tokenizer.chat([{"role":"user","content":prompt}]) if chat else prompt)
        start = time.monotonic()
        self.prefill(ids)
        prefill = time.monotonic()-start
        generated, times = [], []
        for _ in range(min(max_new_tokens, self.max_seq-len(ids))):
            token = int(self.logits[0].argmax())
            generated.append(token)
            if on_token:
                on_token(token, self.tokenizer.decode([token]))
            if token in self.tokenizer.special_ids:
                break
            start = time.monotonic()
            self.decode(token)
            times.append(time.monotonic()-start)
        return generated, self.tokenizer.decode(generated), {"prefill_seconds":prefill,"per_token_seconds":times}


class _PhysicalStats:
    def __init__(self, model):
        self.model = model

    @property
    def stats(self):
        return {k: v for k, v in self.model.stats().items() if k in ('calls','jobs','cycles','macs')}


def __getattr__(name):
    # Saved notebooks refer to these old names; all new callers use the names above.
    aliases = {'OfficialEngine':ONNXReferenceEngine, 'CompiledOfficialEngine':LLMEngine}
    if name in aliases:
        return aliases[name]
    raise AttributeError(name)
