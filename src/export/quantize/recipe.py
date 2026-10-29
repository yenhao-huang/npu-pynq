"""Resolve a QuantRecipe against one model's dimensions."""

from __future__ import annotations


def choose_group(requested: int, dims: list[int]) -> int:
    """The INT4 group size: ``requested`` if it divides every K, else the
    largest of 128/64/32 that does (SmolLM's hidden size is 576)."""
    for g in (requested, 128, 64, 32, 16):
        if g and all(d % g == 0 for d in dims):
            return g
    raise ValueError(f"no INT4 group size divides {dims}")
