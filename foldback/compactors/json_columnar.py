"""Lossless JSON-array compaction.

The single biggest win in agent tool output is an array of similarly-shaped
objects (API responses, DB rows, search hits). Serialized as JSON, every
row repeats every key:

    [{"id": 1, "name": "a", "status": "ok"},
     {"id": 2, "name": "b", "status": "ok"}, ...]

Rendered columnar, the keys are written once:

    cols: id | name | status
    1 | a | ok
    2 | b | ok

No data is dropped. Missing keys become a null marker; nested values are
kept as compact JSON in the cell. The header tells the model exactly how to
read each row, so it is fully reconstructable.
"""

from __future__ import annotations

import json
from typing import Any

# Sentinel for a key absent from a given row. Distinct from JSON null so the
# rendering is reversible: "∅" means "key not present", "null" means present
# with value null.
_ABSENT = "∅"
_SEP = " | "


def _is_uniform_object_array(value: Any) -> bool:
    """True if ``value`` is a list of >=2 flat-ish JSON objects."""
    if not isinstance(value, list) or len(value) < 2:
        return False
    return all(isinstance(item, dict) for item in value)


def _ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Union of keys across all rows, preserving first-seen order."""
    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    return cols


def _cell(value: Any) -> str:
    """Render one cell value losslessly as a single line."""
    if isinstance(value, str):
        # Escape the separator and newlines so the row stays parseable.
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")
    # Numbers, bools, null, and nested structures -> compact JSON.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_json(text: str, parsed: Any) -> str:
    """Return a columnar rendering of ``parsed`` if it shrinks, else ``text``.

    ``parsed`` is the already-decoded JSON (callers parse once and pass it
    in). Only uniform object-arrays are compacted; anything else is returned
    untouched.
    """
    if not _is_uniform_object_array(parsed):
        return text

    rows: list[dict[str, Any]] = parsed
    cols = _ordered_columns(rows)
    if not cols:
        return text

    header = "cols: " + _SEP.join(cols)
    lines = [header]
    for row in rows:
        cells = [
            _cell(row[col]) if col in row else _ABSENT
            for col in cols
        ]
        lines.append(_SEP.join(cells))

    rendered = (
        "[foldback:columnar rows=%d cols=%d; '%s'=absent]\n%s"
        % (len(rows), len(cols), _ABSENT, "\n".join(lines))
    )
    return rendered if len(rendered) < len(text) else text
