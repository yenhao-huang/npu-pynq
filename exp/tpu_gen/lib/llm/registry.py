"""Backend registry.

A backend is registered under a short name with a factory that imports its
module lazily, so selecting `replay` never needs the `anthropic` package and
selecting `claude` never needs `openai`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from tpugen_types import FlowError

from .base import LLMBackend

Factory = Callable[..., LLMBackend]

_BACKENDS: dict[str, Factory] = {}
_DESCRIPTIONS: dict[str, str] = {}


def register(name: str, factory: Factory, description: str = "") -> None:
    """Add a backend under `name`, replacing any previous registration."""
    _BACKENDS[name] = factory
    _DESCRIPTIONS[name] = description


def available_backends() -> list[str]:
    """Registered names, in registration order."""
    return list(_BACKENDS)


def describe_backends() -> dict[str, str]:
    """Registered names mapped to their one-line description."""
    return dict(_DESCRIPTIONS)


def get_backend(name: str, **kwargs) -> LLMBackend:
    """Build a backend by registered name."""
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise FlowError(
            f"unknown LLM backend {name!r}; available: "
            + ", ".join(available_backends())
        ) from None
    return factory(**kwargs)


def _replay(**kwargs) -> LLMBackend:
    from paths import TRAIN_DATASET
    from .replay import ReplayBackend

    return ReplayBackend(Path(kwargs.pop("dataset", TRAIN_DATASET)))


def _anthropic(**kwargs) -> LLMBackend:
    from .api import AnthropicBackend

    return AnthropicBackend(**kwargs)


def _openai(**kwargs) -> LLMBackend:
    from .api import OpenAIBackend

    return OpenAIBackend(**kwargs)


def _claude(**kwargs) -> LLMBackend:
    from .cli import ClaudeCLIBackend

    return ClaudeCLIBackend(**kwargs)


def _codex(**kwargs) -> LLMBackend:
    from .cli import CodexCLIBackend

    return CodexCLIBackend(**kwargs)


register("replay", _replay, "nearest neighbour over the training set, no network")
register("claude", _claude, "Claude Code CLI, one-shot (`claude -p`)")
register("codex", _codex, "OpenAI Codex CLI, one-shot (`codex exec`)")
register("anthropic", _anthropic, "Anthropic API, needs ANTHROPIC_API_KEY")
register("openai", _openai, "OpenAI API, needs OPENAI_API_KEY")
