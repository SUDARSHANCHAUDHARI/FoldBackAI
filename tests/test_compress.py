"""Behavioral tests for FoldBack.

The two invariants that matter: (1) the cache hot zone is never mutated, and
(2) a transform is applied only when it actually saves tokens.
"""

from __future__ import annotations

import json

from foldback import CompressConfig, compress
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
    assert live_zone_start(messages) == 1


def test_frozen_messages_are_same_objects():
    frozen = {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
    live = {"role": "user", "content": "  trailing   \n\n\n\n  done"}
    result = compress([frozen, live])
    # Identity, not equality: the hot-zone object is forwarded as-is.
    assert result.messages[0] is frozen


# ── detection ────────────────────────────────────────────────────────────

def test_detect_json():
    assert detect('[{"a":1}]') is ContentType.JSON


def test_detect_log():
    text = "\n".join(f"2026-06-19T12:00:0{i} INFO request handled" for i in range(5))
    assert detect(text) is ContentType.LOG


def test_detect_text():
    assert detect("just some prose about a thing") is ContentType.TEXT


def test_detect_log_with_leading_ansi():
    text = "\n".join("\x1b[33mWARN retry\x1b[0m" for _ in range(5))
    assert detect(text) is ContentType.LOG


# ── json columnar ────────────────────────────────────────────────────────

def test_json_columnar_shrinks_uniform_array():
    rows = [{"id": i, "name": f"n{i}", "status": "ok"} for i in range(20)]
    text = json.dumps(rows)
    out = compact_json(text, rows)
    assert len(out) < len(text)
    assert out.startswith("<<fb:columnar")


def test_json_mixed_schema_is_passthrough():
    # Rows with different key sets cannot map back unambiguously -> untouched.
    rows = [{"a": i, "b": i} if i % 2 == 0 else {"a": i} for i in range(20)]
    text = json.dumps(rows)
    assert compact_json(text, rows) == text


def test_json_scalar_array_untouched():
    text = json.dumps([1, 2, 3])
    assert compact_json(text, [1, 2, 3]) == text


# ── logs (normalizing) ───────────────────────────────────────────────────

def test_log_dedup_preserves_count():
    text = "boot\n" + "retry\n" * 5 + "done"
    out = compact_log(text)
    assert "(x5)" in out
    assert "boot" in out and "done" in out


def test_log_strips_ansi():
    out = compact_log("\x1b[31mERROR\x1b[0m boom")
    assert "\x1b[" not in out
    assert "ERROR boom" in out


# ── text (normalizing) ───────────────────────────────────────────────────

def test_text_collapses_blank_runs():
    assert compact_text("a\n\n\n\n\nb") == "a\n\nb"


def test_text_strips_trailing_whitespace():
    assert compact_text("a   \nb") == "a\nb"


# ── end to end ───────────────────────────────────────────────────────────

def test_compress_saves_tokens_on_json_tool_result():
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


def test_compress_rejects_non_list():
    import pytest

    with pytest.raises(TypeError):
        compress("not a list")  # type: ignore[arg-type]


def test_compress_min_savings_gate_blocks_marginal_wins():
    # A tiny array whose columnar form barely differs: a high threshold
    # should reject it.
    rows = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]
    messages = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": json.dumps(rows)},
    ]
    strict = compress(messages, config=CompressConfig(min_savings=0.9))
    assert strict.transforms == []


def test_compress_disabled_transform_is_skipped():
    rows = [{"id": i, "name": f"n{i}"} for i in range(30)]
    messages = [
        {"role": "user", "content": "list"},
        {"role": "tool", "content": json.dumps(rows)},
    ]
    cfg = CompressConfig(enabled=frozenset())  # nothing enabled
    result = compress(messages, config=cfg)
    assert result.transforms == []
    assert result.tokens_saved == 0


def test_compress_handles_content_block_list():
    rows = [{"id": i, "k": "v"} for i in range(30)]
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": json.dumps(rows)},
            ],
        },
    ]
    result = compress(messages)
    assert "json:columnar" in result.transforms
