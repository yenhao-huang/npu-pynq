"""LLM backends. Every backend answers generate(prompt) -> .vh text."""

from .base import LLMBackend, clean_vh
from .replay import ReplayBackend
from .api import AnthropicBackend, OpenAIBackend
from .cli import CLIAgentBackend, ClaudeCLIBackend, CodexCLIBackend
from .registry import (
    available_backends,
    describe_backends,
    get_backend,
    register,
)

__all__ = [
    "LLMBackend", "ReplayBackend", "AnthropicBackend", "OpenAIBackend",
    "CLIAgentBackend", "ClaudeCLIBackend", "CodexCLIBackend",
    "clean_vh", "get_backend", "register", "available_backends",
    "describe_backends",
]
