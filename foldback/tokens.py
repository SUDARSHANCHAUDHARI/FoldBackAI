"""Token counting.

Two backends:

* **Heuristic** (default, zero-dependency): ``len(text) / 4``. Good enough to
  drive the "did this actually shrink?" gate and to report rough savings.
* **Exact** (optional): if ``tiktoken`` is installed, ``estimate`` uses the
  encoding that matches ``model``. Install with ``pip install foldback[exact]``.

The backend is chosen per call from the ``model`` argument, so callers that
pass a model get exact counts when ``tiktoken`` is present and fall back to
the heuristic otherwise — never an import error.
"""

from __future__ import annotations

import functools

_CHARS_PER_TOKEN = 4.0


def _heuristic(text: str) -> int:
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


@functools.lru_cache(maxsize=8)
def _encoding(model: str):  # type: ignore[no-untyped-def]
    """Return a cached tiktoken encoding for ``model``, or None if unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model name — use the modern default encoding.
        return tiktoken.get_encoding("o200k_base")


def estimate(text: str, model: str | None = None) -> int:
    """Count tokens in ``text``.

    With ``model`` set and ``tiktoken`` installed, returns an exact count for
    that model's encoding. Otherwise falls back to the char/4 heuristic.
    """
    if not text:
        return 0
    if model is not None:
        enc = _encoding(model)
        if enc is not None:
            return len(enc.encode(text))
    return _heuristic(text)
