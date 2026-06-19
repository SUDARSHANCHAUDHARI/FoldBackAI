"""The one-function API: ``compress(messages)``.

Pipeline per live-zone string:
  1. Detect content type (json / log / text).
  2. Apply the matching lossless compactor.
  3. Keep the result only if it actually shrank (compactors self-gate too).

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


def _compact_string(text: str) -> tuple[str, str | None]:
    """Run the right lossless compactor. Returns (result, transform_name)."""
    kind = detect(text)
    if kind is ContentType.JSON:
        parsed = parse_json(text)
        if parsed is not None:
            out = compact_json(text, parsed)
            return out, ("json:columnar" if out != text else None)
    if kind is ContentType.LOG:
        out = compact_log(text)
        return out, ("log:dedup" if out != text else None)
    out = compact_text(text)
    return out, ("text:whitespace" if out != text else None)


def _compact_content(content: Any, transforms: list[str]) -> Any:
    """Compress a message ``content`` field (str or content-block list)."""
    if isinstance(content, str):
        out, name = _compact_string(content)
        if name:
            transforms.append(name)
        return out

    if isinstance(content, list):
        new_blocks = []
        changed = False
        for block in content:
            text = _block_text(block)
            if text is None:
                new_blocks.append(block)
                continue
            out, name = _compact_string(text)
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
    # Anthropic text block / tool_result with string content.
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return block["text"]
    if block.get("type") == "tool_result" and isinstance(block.get("content"), str):
        return block["content"]
    return None


def _set_block_text(block: dict, value: str) -> dict:
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


def compress(messages: list[Message]) -> CompressResult:
    """Losslessly compact the live zone of ``messages``.

    Returns a ``CompressResult``. The returned ``messages`` list is safe to
    send to any provider: frozen (cache hot zone) messages are the identical
    objects you passed in; only live-zone content is rewritten, and only
    when a lossless transform shrank it.
    """
    if not messages:
        return CompressResult([], 0, 0)

    start = live_zone_start(messages)
    original_tokens = sum(tokens.estimate(_message_text(m)) for m in messages)

    out: list[Message] = list(messages[:start])  # frozen: same objects
    transforms: list[str] = []

    for message in messages[start:]:
        content = message.get("content")
        new_content = _compact_content(content, transforms)
        if new_content is content:
            out.append(message)  # unchanged: same object
        else:
            out.append({**message, "content": new_content})

    compressed_tokens = sum(tokens.estimate(_message_text(m)) for m in out)
    return CompressResult(out, original_tokens, compressed_tokens, transforms)
