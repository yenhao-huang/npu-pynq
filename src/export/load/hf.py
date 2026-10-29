"""Find a Hugging Face model: a local path or the newest cached snapshot."""

from __future__ import annotations

from pathlib import Path


def resolve_hf(model_id_or_path: str) -> Path:
    """A local directory, or the newest snapshot of a model in the HF cache."""
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path
    import os
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshots = cache / f"models--{model_id_or_path.replace('/', '--')}" / "snapshots"
    candidates = sorted(snapshots.glob("*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"{model_id_or_path} is neither a path nor in {cache}")
    return candidates[-1]
