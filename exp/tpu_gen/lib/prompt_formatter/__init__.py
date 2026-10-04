"""User prompt -> the TPU-Gen prompt format the LLM was trained on.

The training data (src_code/beta_train_all.json) pairs a Description/Metrics
pair with a .vh file, so the LLM only behaves well when the prompt matches
that shape exactly.
"""

from .formatter import (
    AP_ADDERS,
    AP_MULTIPLIERS,
    DesignSpec,
    format_prompt,
    parse_user_prompt,
)

__all__ = [
    "AP_ADDERS",
    "AP_MULTIPLIERS",
    "DesignSpec",
    "format_prompt",
    "parse_user_prompt",
]
