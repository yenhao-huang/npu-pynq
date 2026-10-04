"""`.vh` -> the .v files the design needs.

Deliberately does not reuse the upstream RAG.ipynb matcher, which compared
every `define name (including DW, M, N, VBL) against filenames as a
substring and pulled in unrelated files.
"""

from .graph import RTLLibrary, RetrievalResult, retrieve, selector_macros
from .preprocess import collect_defines, resolve, strip_comments

__all__ = [
    "RTLLibrary", "RetrievalResult", "retrieve", "selector_macros",
    "collect_defines", "resolve", "strip_comments",
]
