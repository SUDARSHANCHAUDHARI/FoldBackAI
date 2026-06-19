"""Lossless log compaction.

Logs waste tokens two ways: consecutive identical lines, and ANSI color
codes. Both are removed without losing information — repeated lines collapse
to a single line plus an explicit ``(xN)`` count, so the data (this line
occurred N times, in this position) is preserved, not dropped.
"""

from __future__ import annotations

import re

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def compact_log(text: str) -> str:
    """Strip ANSI codes and run-length-encode consecutive duplicate lines."""
    cleaned = _ANSI.sub("", text)
    lines = cleaned.split("\n")

    out: list[str] = []
    prev: str | None = None
    count = 0

    def flush() -> None:
        if prev is None:
            return
        out.append(prev if count == 1 else f"{prev}  (x{count})")

    for line in lines:
        if line == prev:
            count += 1
        else:
            flush()
            prev = line
            count = 1
    flush()

    rendered = "\n".join(out)
    return rendered if len(rendered) < len(text) else text
