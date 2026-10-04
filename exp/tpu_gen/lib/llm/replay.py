from __future__ import annotations

import json
import re
from pathlib import Path

from tpugen_types import FlowError

from .base import LLMBackend, clean_vh

_WORD = re.compile(r"[A-Za-z_]+|\d+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class ReplayBackend(LLMBackend):
    """Nearest-neighbour lookup over the TPU-Gen training set.

    Not a language model: it returns the .vh of the closest description in
    beta_train_all.json. It exists so the rest of the flow can be exercised
    end to end without a GPU or an API key, and as a regression baseline for
    the real backends.
    """

    name = "replay"

    def __init__(self, dataset: Path):
        if not dataset.exists():
            raise FlowError(f"replay dataset not found: {dataset}")
        self.dataset = dataset
        raw = json.loads(dataset.read_text(encoding="utf-8-sig"))
        self.entries = [
            {
                "description": e["description"],
                "code": e["code"],
                "metrics": e.get("metrics", {}),
                "tokens": _tokens(e["description"]),
            }
            for e in raw
        ]
        if not self.entries:
            raise FlowError(f"replay dataset is empty: {dataset}")

    def generate(self, prompt: str) -> str:
        want = _tokens(prompt)
        best = max(
            self.entries,
            key=lambda e: len(want & e["tokens"]) / len(want | e["tokens"]),
        )
        return clean_vh(best["code"])
