"""Fields shared by every category input."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Backendable(BaseModel):
    """Base input: an agent names the category, not the tool behind it.

    `backend` exists so a human or a test can pin an implementation. Agents are
    expected to leave it unset and get the category default.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str | None = Field(
        default=None,
        description="Override the default backend for this category. Leave unset unless you need a specific tool.",
    )
