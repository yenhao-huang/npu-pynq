"""transformers: ``AutoModelForCausalLM`` from a local directory or a Hub id."""

from __future__ import annotations

from pathlib import Path

import torch


def load(source: str | Path, **kwargs) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.float32, **kwargs)
    return model.eval().requires_grad_(False)


def files(source: str | Path) -> dict[str, Path]:
    """Files the runtime needs beside the compiled model (the tokenizer)."""
    root = Path(source)
    return {name: root / name for name in ('tokenizer.json', 'tokenizer_config.json')
            if (root / name).is_file()}
