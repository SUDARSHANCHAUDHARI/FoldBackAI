"""Live-zone detection.

The single rule that keeps provider prompt caches alive: never modify
anything in the cache hot zone. The hot zone is the stable prefix —
system prompt, tool definitions, and every message up to and including
the last ``cache_control`` breakpoint. Everything after that is the
*live zone*, and only the live zone is eligible for compression.

If the message list uses no ``cache_control`` markers, we fall back to
treating only the final message as live (the safest possible default —
the tail changes every turn anyway, so compressing it costs no cache).
"""

from __future__ import annotations

from typing import Any

Message = dict[str, Any]


def _has_cache_control(message: Message) -> bool:
    """True if ``message`` carries a cache_control marker anywhere."""
    if message.get("cache_control") is not None:
        return True
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("cache_control") is not None:
                return True
    return False


def live_zone_start(messages: list[Message]) -> int:
    """Return the index of the first live-zone message.

    Everything at index ``< start`` is frozen (cache hot zone) and must be
    forwarded byte-for-byte. Everything at ``>= start`` may be compressed.
    """
    if not messages:
        return 0

    last_marker = -1
    for i, message in enumerate(messages):
        if _has_cache_control(message):
            last_marker = i

    if last_marker >= 0:
        # Freeze through the marker; live zone begins right after it.
        return min(last_marker + 1, len(messages))

    # No markers: only the final message is live.
    return len(messages) - 1
