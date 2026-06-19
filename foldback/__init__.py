"""FoldBack — lossless context compression for LLM agents.

Fold it down, fold it back. FoldBack shrinks the *live zone* of a message
list (the latest tool result / latest user message) before it reaches the
model, using only lossless, data-preserving transforms:

    from foldback import compress

    result = compress(messages)
    result.messages          # same format, fewer tokens
    result.tokens_saved      # estimated tokens saved
    result.ratio             # compressed / original (0.4 == 60% saved)

Design rules (see README):
  1. Passthrough is sacred. Anything before the last ``cache_control``
     breakpoint is never touched, so provider prompt caches keep hitting.
  2. Only the live zone is rewritten — the latest user message and the
     latest tool result.
  3. Transforms are lossless. No row is dropped; no value is discarded.
     Rendering changes (JSON array -> columnar), data does not.
  4. If a transform does not shrink the content, the original is returned
     unchanged.
"""

from __future__ import annotations

from .compactors import is_columnar, restore_columnar
from .compress import CompressConfig, CompressResult, compress

__all__ = [
    "compress",
    "CompressConfig",
    "CompressResult",
    "restore_columnar",
    "is_columnar",
]
__version__ = "0.1.1"
