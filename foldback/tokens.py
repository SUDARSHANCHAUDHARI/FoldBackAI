"""Token estimation.

Deliberately dependency-free. A char/4 heuristic is close enough to drive
the "did this actually shrink?" gate and to report savings. If you need
exact counts, swap in a real tokenizer behind ``estimate``.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4.0


def estimate(text: str) -> int:
    """Estimate the token count of ``text`` (~chars / 4)."""
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))
