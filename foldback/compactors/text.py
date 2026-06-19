"""Lossless plain-text compaction.

Conservative on purpose: prose is where lossy compression hurts, so we only
remove whitespace that carries no information — trailing spaces and runs of
3+ blank lines collapsed to one. Words, punctuation, and single/double line
breaks are untouched.
"""

from __future__ import annotations

import re

_TRAILING_WS = re.compile(r"[ \t]+(?=\n)")
_BLANK_RUN = re.compile(r"\n{3,}")


def compact_text(text: str) -> str:
    """Trim trailing whitespace and collapse 3+ blank lines to one blank."""
    out = _TRAILING_WS.sub("", text)
    out = _BLANK_RUN.sub("\n\n", out)
    return out if len(out) < len(text) else text
