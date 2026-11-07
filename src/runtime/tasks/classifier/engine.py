"""Image classification through a classifier package (manifest key ``classifier``)."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from src.runtime.share.loader import ExportedPackage


class ImageEngine:
    """Image classification through an pynqo-package/2 package."""

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
