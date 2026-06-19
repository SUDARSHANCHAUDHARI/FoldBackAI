"""Lossless compactors.

Each compactor takes a raw string and returns a (possibly) smaller string
that preserves all the data. None of them drop rows, keys, or values — they
only change *representation*. If a compactor cannot improve on the input it
returns the input unchanged.
"""

from __future__ import annotations

from .json_columnar import compact_json
from .logs import compact_log
from .text import compact_text

__all__ = ["compact_json", "compact_log", "compact_text"]
