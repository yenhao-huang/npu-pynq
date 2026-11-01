"""Run a package exported through PyTorch (format npu-package/2).

The package holds model.so per CPU model, weights.bin and, for each delegate
device, its own artefacts (programs.* for the NPU). Entry points take
memref descriptors through MLIR's C interface:

    _mlir_ciface_<entry>(ws, *inputs, *outputs)

An LLM's KV cache is functional: ``decode``/``prefill`` read one cache set
and write the next, so the engine keeps two sets and swaps them per call.
"""

from __future__ import annotations

import ctypes
import json
import platform
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from src.runtime.core.sampling import Sampler, SamplingParams
from src.runtime.core.tokenizer import Tokenizer
from src.runtime.delegate import Dispatcher

FORMAT = 'npu-package/2'
DTYPES = {'f32': np.float32, 'i64': np.int64, 'i32': np.int32, 'i8': np.int8}


def descriptor_type(rank: int) -> type:
    fields = [('allocated', ctypes.c_void_p), ('aligned', ctypes.c_void_p), ('offset', ctypes.c_ssize_t)]
    if rank:
        fields += [('sizes', ctypes.c_ssize_t * rank), ('strides', ctypes.c_ssize_t * rank)]
    return type(f'MemRef{rank}D', (ctypes.Structure,), {'_fields_': fields})


def descriptor(array: np.ndarray) -> ctypes.Structure:
    if not array.flags.c_contiguous:
        raise ValueError('memref arguments must be C-contiguous')
    desc = descriptor_type(array.ndim)()
    desc.allocated = desc.aligned = array.ctypes.data
    desc.offset = 0
    for i, (size, stride) in enumerate(zip(array.shape, array.strides)):
        desc.sizes[i] = size
        desc.strides[i] = stride // array.itemsize
    return desc


def cpu_model() -> str:
    machine = platform.machine()
    return 'arm-cortex-a9' if machine.startswith('armv7') else machine


def on_board() -> bool:
    return Path('/etc/profile.d/pynq_venv.sh').exists()


class ExportedPackage:
    """model.so with its weights and delegate backends, ready to call."""

    def __init__(self, root: str | Path, overlay: Any = None) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('format') != FORMAT:
            raise ValueError(f'{self.root}: not an {FORMAT} package')
        self.dispatcher = Dispatcher()
        self.backends = {}
        devices = {d['device'] for d in self.manifest['delegates']}
        if 'npu' in devices:
            from src.runtime.delegate.npu import NpuBackend
            self.backends['npu'] = NpuBackend(self.root, overlay)
        self.dispatcher.install(self.manifest['delegates'],
                                {name: b.execute for name, b in self.backends.items()})
        libraries = self.manifest['libraries']
        model = cpu_model()
        name = next((k for k in libraries if k == model or k.startswith(model + '-')), None)
        if name is None:
            raise ValueError(f'no library for this CPU ({model}); package has {sorted(libraries)}')
        if model == 'arm-cortex-a9':
            for dep in ('libm.so.6', 'libgcc_s.so.1'):
                ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(str(self.root / libraries[name]))
        weights = self.root / 'weights.bin'
        self.weights = np.memmap(weights, dtype=np.uint8, mode='r') if weights.stat().st_size else np.zeros(8, np.uint8)

    def call(self, entry: str, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        fn = getattr(self.lib, f'_mlir_ciface_{entry}')
        descs = [descriptor(a) for a in (self.weights, *inputs, *outputs)]
        fn.restype = None
        fn(*(ctypes.byref(d) for d in descs))
        self.dispatcher.check()

    def stats(self) -> dict[str, int]:
        npu = self.backends.get('npu')
        stats = npu.stats() if npu else {}
        stats['delegate_calls'] = self.dispatcher.calls
        return stats

    def close(self) -> None:
        """Release the backends' board buffers; the package cannot be called afterwards."""
        for backend in self.backends.values():
            backend.close()
        self.backends = {}


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


class DelegatedImageEngine:
    """Image classification through an npu-package/2 package (ImageEngine's interface)."""

    def __init__(self, package: str | Path, backend: str | None = None, *, overlay: Any = None,
                 stream_weights: Any = None) -> None:
        if backend == 'pynq' and overlay is None:
            raise ValueError('the pynq backend needs the loaded overlay')
        self.package = ExportedPackage(package, overlay if backend != 'sim' else None)
        meta = self.package.manifest['classifier']
        self.contract = meta['input']
        self.classes = int(meta['classes'])
        self.last_seconds = 0.0
        self.last_stats = {}

    def predict(self, image) -> np.ndarray:
        image = np.ascontiguousarray(image)
        if image.dtype != np.float32 or list(image.shape) != self.contract['shape']:
            raise ValueError('Expected a normalized float32 image matching the package input')
        if not np.isfinite(image).all():
            raise ValueError('Image contains nonfinite values')
        logits = np.zeros((1, self.classes), np.float32)
        before = self.package.stats()
        start = time.perf_counter()
        self.package.call('forward', [image], [logits])
        self.last_seconds = time.perf_counter() - start
        after = self.package.stats()
        self.last_stats = {k: after[k] - before.get(k, 0) for k in ('calls', 'jobs', 'cycles', 'macs', 'polls')
                           if k in after}
        if not np.isfinite(logits).all():
            raise RuntimeError('Compiled image model returned nonfinite logits')
        return logits[0]

    def close(self) -> None:
        self.package.close()
