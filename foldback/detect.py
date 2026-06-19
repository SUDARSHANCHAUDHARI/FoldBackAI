"""Cheap content-type detection.

No ML model, no downloads — just fast heuristics good enough to route a
string to the right lossless compactor. The order matters: JSON is checked
by actually parsing (cheap and unambiguous), then logs, then plain text.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

# Strip ANSI color codes before the per-line log check — colored logs lead
# with "\x1b[..m" which would otherwise hide the level/timestamp marker.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# A line that begins with a timestamp / log-level marker. Covers the common
# shapes: ISO-8601, "2026-06-19 12:00:00", "[INFO]", "ERROR:", syslog, etc.
_LOG_LINE = re.compile(
    r"""^\s*(
        \d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}      # 2026-06-19T12:00:00
      | \[?(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\]?\b  # levels
      | \w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}          # syslog: Jun 19 12:00:00
    )""",
    re.IGNORECASE | re.VERBOSE,
)


class ContentType(str, Enum):
    JSON = "json"
    LOG = "log"
    TEXT = "text"


def parse_json(text: str) -> Any | None:
    """Return parsed JSON if ``text`` is a single JSON document, else None."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return None


def detect(text: str) -> ContentType:
    """Classify ``text`` into a content type for routing."""
    if parse_json(text) is not None:
        return ContentType.JSON

    lines = [ln for ln in _ANSI.sub("", text).splitlines() if ln.strip()]
    if len(lines) >= 3:
        log_lines = sum(1 for ln in lines if _LOG_LINE.match(ln))
        if log_lines / len(lines) >= 0.6:
            return ContentType.LOG

    return ContentType.TEXT
