"""The one-function API: ``compress(messages, model=...)``.

Pipeline per live-zone string:
  1. Detect content type (json / log / text).
  2. Apply the matching compactor.
  3. Keep the result only if it actually saves tokens (>= ``min_savings``).

Frozen messages (the cache hot zone) are returned as the *same objects*
that came in — never copied, never re-serialized — so nothing can drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import tokens
from .compactors import compact_json, compact_log, compact_text
from .detect import ContentType, detect, parse_json
from .livezone import live_zone_start

Message = dict[str, Any]

# Transform names, also used as config toggles.
JSON_COLUMNAR = "json:columnar"
LOG_DEDUP = "log:dedup"
TEXT_WHITESPACE = "text:whitespace"
_ALL_TRANSFORMS = frozenset({JSON_COLUMNAR, LOG_DEDUP, TEXT_WHITESPACE})


@dataclass(frozen=True)
class CompressConfig:
    """Options for ``compress``.

    Attributes:
        model: provider model id. When set and ``tiktoken`` is installed,
            token counts (and the savings gate) become exact.
        min_savings: a transform's output is kept only if it saves at least
            this fraction of the block's tokens. ``0.0`` keeps any net win.
        enabled: which transforms may run. Defaults to all.
    """

    model: str | None = None
    min_savings: float = 0.0
    enabled: frozenset[str] = _ALL_TRANSFORMS


@dataclass
class CompressResult:
    """Outcome of a ``compress`` call."""

    messages: list[Message]
    original_tokens: int
    compressed_tokens: int
    transforms: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def ratio(self) -> float:
        """compressed / original. 0.4 means 60% saved. 1.0 means no change."""
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


def _gated(original: str, candidate: str, name: str, cfg: CompressConfig) -> str | None:
    """Return ``name`` if ``candidate`` clears the token-savings gate, else None.

    The gate is measured in tokens (not bytes) because tokens are what the
    model and the bill care about — a byte-smaller rendering that tokenizes
    larger is rejected.
    """
    if candidate == original:
        return None
    before = tokens.estimate(original, cfg.model)
    after = tokens.estimate(candidate, cfg.model)
    if before == 0:
        return None
    if (before - after) / before >= cfg.min_savings and after < before:
        return name
    return None


def _compact_string(text: str, cfg: CompressConfig) -> tuple[str, str | None]:
    """Run the right compactor under ``cfg``. Returns (result, transform_name)."""
    kind = detect(text)

    if kind is ContentType.JSON and JSON_COLUMNAR in cfg.enabled:
        parsed = parse_json(text)
        if parsed is not None:
            out = compact_json(text, parsed)
            name = _gated(text, out, JSON_COLUMNAR, cfg)
            if name:
                return out, name

    if kind is ContentType.LOG and LOG_DEDUP in cfg.enabled:
        out = compact_log(text)
        name = _gated(text, out, LOG_DEDUP, cfg)
        if name:
            return out, name

    if kind is ContentType.TEXT and TEXT_WHITESPACE in cfg.enabled:
        out = compact_text(text)
        name = _gated(text, out, TEXT_WHITESPACE, cfg)
        if name:
            return out, name

    return text, None


def _compact_content(content: Any, transforms: list[str], cfg: CompressConfig) -> Any:
    """Compress a message ``content`` field (str or content-block list)."""
    if isinstance(content, str):
        out, name = _compact_string(content, cfg)
        if name:
            transforms.append(name)
            return out
        return content

    if isinstance(content, list):
        new_blocks: list[Any] = []
        changed = False
        for block in content:
            text = _block_text(block)
            if text is None:
                new_blocks.append(block)
                continue
            out, name = _compact_string(text, cfg)
            if name:
                transforms.append(name)
                new_blocks.append({**block, **_set_block_text(block, out)})
                changed = True
            else:
                new_blocks.append(block)
        return new_blocks if changed else content

    return content


def _block_text(block: Any) -> str | None:
    """Extract the compressible text from a content block, if any."""
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "text":
        text = block.get("text")
        return text if isinstance(text, str) else None
    if btype == "tool_result":
        content = block.get("content")
        return content if isinstance(content, str) else None
    return None


def _set_block_text(block: dict[str, Any], value: str) -> dict[str, str]:
    """Return the field overlay that writes ``value`` back into ``block``."""
    if block.get("type") == "text":
        return {"text": value}
    return {"content": value}


def _message_text(message: Message) -> str:
    """Best-effort flatten of a message's content for token counting."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_block_text(b) or "" for b in content]
        return "\n".join(parts)
    return ""


def compress(
    messages: list[Message],
    model: str | None = None,
    *,
    config: CompressConfig | None = None,
) -> CompressResult:
    """Compact the live zone of ``messages``.

    Args:
        messages: a provider-style message list (dicts with ``role`` and
            ``content``). The list and its frozen messages are never mutated.
        model: convenience shortcut for ``CompressConfig(model=...)``. Ignored
            if ``config`` is given.
        config: full options. Overrides ``model`` when both are provided.

    Returns:
        A ``CompressResult``. The returned ``messages`` are safe to send to
        any provider: frozen (cache hot zone) messages are the identical
        objects you passed in; only live-zone content is rewritten, and only
        when a transform cleared the token-savings gate.

    Raises:
        TypeError: if ``messages`` is not a list.
    """
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    if config is None:
        config = CompressConfig(model=model)

    if not messages:
        return CompressResult([], 0, 0)

    start = live_zone_start(messages)
    original_tokens = sum(tokens.estimate(_message_text(m), config.model) for m in messages)

    out: list[Message] = list(messages[:start])  # frozen: same objects
    transforms: list[str] = []

    for message in messages[start:]:
        if not isinstance(message, dict):
            out.append(message)  # forward anything unexpected untouched
            continue
        content = message.get("content")
        new_content = _compact_content(content, transforms, config)
        if new_content is content:
            out.append(message)
        else:
            out.append({**message, "content": new_content})

    compressed_tokens = sum(tokens.estimate(_message_text(m), config.model) for m in out)
    return CompressResult(out, original_tokens, compressed_tokens, transforms)
