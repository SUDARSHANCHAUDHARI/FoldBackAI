"""Content-preserving compactors.

Two categories, with different guarantees:

* **Reversible** — ``compact_json`` (columnar). Exact round-trip: every row,
  key, value, and type is reconstructable via ``restore_columnar``.
* **Normalizing** — ``compact_log`` and ``compact_text``. They remove only
  non-semantic bytes (ANSI color codes, trailing whitespace, blank-line runs)
  and collapse *consecutive identical* log lines into one line plus an
  explicit ``(xN)`` count. No textual content is dropped, but these are not
  byte-for-byte reversible (the stripped formatting is gone for good).

Every compactor self-gates: if it cannot shrink the input, it returns the
input unchanged.
"""

from __future__ import annotations

from .json_columnar import compact_json, is_columnar, restore_columnar
from .logs import compact_log
from .text import compact_text

__all__ = [
    "compact_json",
    "compact_log",
    "compact_text",
    "is_columnar",
    "restore_columnar",
]
