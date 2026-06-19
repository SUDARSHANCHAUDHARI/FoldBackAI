"""Proof that the columnar transform is genuinely lossless.

These are the tests that turn "we claim it's reversible" into "we verified it
is." If any of these fail, the core value proposition is broken — that is the
intended signal.
"""

from __future__ import annotations

import json

import pytest

from foldback.compactors import compact_json, is_columnar, restore_columnar

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def _roundtrip(rows: list[dict]) -> None:
    """Compact then restore; assert the data is reconstructed exactly."""
    text = json.dumps(rows)
    out = compact_json(text, rows)
    if not is_columnar(out):
        # Compaction declined (passthrough); nothing to restore.
        assert out == text
        return
    restored = json.loads(restore_columnar(out))
    assert restored == rows


# ── targeted regressions (the bugs the first version had) ─────────────────

def test_string_vs_number_not_confused():
    # "1" (str) and 1 (int) rendered identically in the broken first version.
    rows = [{"v": "1"}, {"v": 1}]  # mixed schema? no — same key, diff types
    # Same key set -> uniform -> compacted; types must survive.
    _roundtrip(rows)


def test_unicode_preserved():
    rows = [{"name": "café", "emoji": "🚀"}, {"name": "naïve", "emoji": "✅"}]
    _roundtrip(rows)


def test_pipe_and_newline_in_values():
    rows = [{"s": "a | b"}, {"s": "line1\nline2"}]
    _roundtrip(rows)


def test_null_vs_missing_distinction():
    # Both rows carry the key (uniform); one value is JSON null.
    rows = [{"x": None}, {"x": 0}]
    _roundtrip(rows)


def test_nested_structures_preserved():
    rows = [
        {"meta": {"a": [1, 2]}, "tags": ["x"]},
        {"meta": {"a": []}, "tags": ["y", "z"]},
    ]
    _roundtrip(rows)


def test_restore_rejects_non_columnar():
    with pytest.raises(ValueError):
        restore_columnar('[{"a":1}]')


# ── property-based: any uniform object array round-trips ───────────────────

_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.text(max_size=40),
)


@st.composite
def _uniform_rows(draw):
    keys = draw(st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=5, unique=True))
    n = draw(st.integers(min_value=2, max_value=15))
    return [{k: draw(_json_scalar) for k in keys} for _ in range(n)]


@settings(max_examples=200)
@given(_uniform_rows())
def test_property_uniform_array_roundtrips(rows):
    _roundtrip(rows)
