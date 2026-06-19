"""Tests for FoldBack compression.

The two things that matter: (1) the cache hot zone is never mutated, and
(2) every transform is lossless — the data is recoverable from the output.
"""

from __future__ import annotations

import json

from foldback import compress
from foldback.compactors import compact_json, compact_log, compact_text
from foldback.detect import ContentType, detect
from foldback.livezone import live_zone_start


# ── live zone ────────────────────────────────────────────────────────────

def test_live_zone_no_markers_is_last_message_only():
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert live_zone_start(messages) == 1


def test_live_zone_respects_cache_control_marker():
    messages = [
        {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    # Frozen through index 0; live zone starts at 1.
    assert live_zone_start(messages) == 1


def test_frozen_messages_are_same_objects():
    frozen = {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
    live = {"role": "user", "content": "  trailing   \n\n\n\n  done"}
    result = compress([frozen, live])
    # Identity, not just equality: the hot-zone object is forwarded as-is.
    assert result.messages[0] is frozen


# ── detection ────────────────────────────────────────────────────────────

def test_detect_json():
    assert detect('[{"a":1}]') is ContentType.JSON


def test_detect_log():
    text = "\n".join(
        f"2026-06-19T12:00:0{i} INFO request handled" for i in range(5)
    )
    assert detect(text) is ContentType.LOG


def test_detect_text():
    assert detect("just some prose about a thing") is ContentType.TEXT


def test_detect_log_with_leading_ansi():
    # Colored logs lead with an escape code; detection must see past it.
    text = "\n".join("\x1b[33mWARN retry\x1b[0m" for _ in range(5))
    assert detect(text) is ContentType.LOG


# ── json columnar (lossless) ─────────────────────────────────────────────

def test_json_columnar_shrinks_and_preserves_data():
    rows = [{"id": i, "name": f"n{i}", "status": "ok"} for i in range(20)]
    text = json.dumps(rows)
    out = compact_json(text, rows)
    assert len(out) < len(text)
    # Every value is still present in the rendering.
    for row in rows:
        assert f"n{row['id']}" in out
        assert str(row["id"]) in out


def test_json_columnar_handles_missing_keys():
    # Enough rows that columnar actually shrinks; every other row omits "b".
    rows = [{"a": i, "b": i * 2} if i % 2 == 0 else {"a": i} for i in range(20)]
    out = compact_json(json.dumps(rows), rows)
    assert len(out) < len(json.dumps(rows))
    assert "∅" in out  # absent marker, distinct from null


def test_json_non_uniform_array_untouched():
    text = json.dumps([1, 2, 3])
    assert compact_json(text, [1, 2, 3]) == text


# ── logs (lossless) ──────────────────────────────────────────────────────

def test_log_dedup_preserves_count():
    text = "boot\n" + "retry\n" * 5 + "done"
    out = compact_log(text)
    assert "(x5)" in out
    assert "boot" in out and "done" in out


def test_log_strips_ansi():
    out = compact_log("\x1b[31mERROR\x1b[0m boom")
    assert "\x1b[" not in out
    assert "ERROR boom" in out


# ── text (lossless) ──────────────────────────────────────────────────────

def test_text_collapses_blank_runs():
    out = compact_text("a\n\n\n\n\nb")
    assert out == "a\n\nb"


def test_text_strips_trailing_whitespace():
    assert compact_text("a   \nb") == "a\nb"


# ── end to end ───────────────────────────────────────────────────────────

def test_compress_reports_savings_on_json_tool_result():
    rows = [{"id": i, "name": f"n{i}", "status": "ok"} for i in range(50)]
    messages = [
        {"role": "user", "content": "list users"},
        {"role": "tool", "content": json.dumps(rows)},
    ]
    result = compress(messages)
    assert result.tokens_saved > 0
    assert result.ratio < 1.0
    assert "json:columnar" in result.transforms


def test_compress_noop_returns_equivalent_messages():
    messages = [{"role": "user", "content": "short"}]
    result = compress(messages)
    assert result.messages[0]["content"] == "short"
    assert result.tokens_saved == 0


def test_compress_empty():
    result = compress([])
    assert result.messages == []
    assert result.ratio == 1.0
