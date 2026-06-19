"""Reversible JSON-array compaction.

The single biggest win in agent tool output is an array of uniformly-shaped
objects (API responses, DB rows, search hits). Serialized as JSON, every
row repeats every key::

    [{"id":1,"name":"a","status":"ok"},{"id":2,"name":"b","status":"ok"}]

FoldBack writes the keys once and each row as a compact JSON *array*::

    <<fb:columnar cols=["id","name","status"] n=2>>
    [1,"a","ok"]
    [2,"b","ok"]

This is **fully reversible**: ``restore_columnar`` reconstructs the exact
original objects. Two correctness rules make that guarantee real:

1. **Values stay JSON.** Each cell is a JSON token, so ``"1"`` (string) and
   ``1`` (number) never collide, unicode is preserved, and nested objects /
   arrays survive untouched.
2. **Uniform schema only.** Every row must share the same key set, so each
   positional array maps back to keys unambiguously. Mixed-schema arrays are
   left untouched (passthrough) rather than compacted lossily.
"""

from __future__ import annotations

import json
from typing import Any

_MARKER_PREFIX = "<<fb:columnar cols="
_MARKER_SUFFIX = ">>"


def _is_object_array(value: Any) -> bool:
    """True if ``value`` is a list of >=2 JSON objects."""
    if not isinstance(value, list) or len(value) < 2:
        return False
    return all(isinstance(item, dict) for item in value)


def _uniform_columns(rows: list[dict[str, Any]]) -> list[str] | None:
    """Return the shared column order if every row has the same key set.

    Uses the first row's key order. Returns ``None`` when rows disagree on
    which keys are present — the signal to skip compaction entirely.
    """
    cols = list(rows[0].keys())
    key_set = set(cols)
    for row in rows[1:]:
        if set(row.keys()) != key_set:
            return None
    return cols


def is_columnar(text: str) -> bool:
    """True if ``text`` is a FoldBack columnar block."""
    return text.startswith(_MARKER_PREFIX)


def compact_json(text: str, parsed: Any) -> str:
    """Return a reversible columnar rendering of ``parsed`` if it shrinks.

    ``parsed`` is the already-decoded JSON (callers parse once and pass it
    in). Only uniform-schema object arrays are compacted; anything else —
    including mixed-schema arrays — is returned unchanged.
    """
    if not _is_object_array(parsed):
        return text

    rows: list[dict[str, Any]] = parsed
    cols = _uniform_columns(rows)
    if not cols:
        return text

    cols_json = json.dumps(cols, ensure_ascii=False, separators=(",", ":"))
    header = f"{_MARKER_PREFIX}{cols_json} n={len(rows)}{_MARKER_SUFFIX}"
    lines = [header]
    for row in rows:
        values = [row[col] for col in cols]
        lines.append(json.dumps(values, ensure_ascii=False, separators=(",", ":")))

    rendered = "\n".join(lines)
    return rendered if len(rendered) < len(text) else text


def restore_columnar(text: str) -> str:
    """Inverse of :func:`compact_json`. Reconstruct the original JSON text.

    Returns the canonical JSON array string for the rows. ``text`` must be a
    columnar block produced by this module (see :func:`is_columnar`).
    """
    if not is_columnar(text):
        raise ValueError("not a FoldBack columnar block")

    header, _, body = text.partition("\n")
    # Header: "<<fb:columnar cols=[...] n=K>>"
    inner = header[len(_MARKER_PREFIX) : -len(_MARKER_SUFFIX)]
    cols_part, _, _ = inner.rpartition(" n=")
    cols: list[str] = json.loads(cols_part)

    rows: list[dict[str, Any]] = []
    for line in body.split("\n") if body else []:
        values = json.loads(line)
        # strict=True turns a corrupted block (wrong value count) into a loud
        # error instead of a silently truncated row.
        rows.append(dict(zip(cols, values, strict=True)))

    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
