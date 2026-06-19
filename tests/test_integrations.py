"""Tests for the SDK wrappers using fake clients (no network, no real SDK)."""

from __future__ import annotations

import json

from foldback.integrations import with_anthropic, with_openai


def _big_rows(n: int = 40) -> str:
    return json.dumps([{"id": i, "name": f"n{i}", "ok": True} for i in range(n)])


# ── fakes mirroring the SDK surface ───────────────────────────────────────

class _FakeMessages:
    def __init__(self, sink: dict) -> None:
        self._sink = sink

    def create(self, **kwargs):
        self._sink.update(kwargs)
        return "anthropic-response"

    def count_tokens(self, **kwargs):  # arbitrary extra method to delegate
        return 123


class _FakeAnthropic:
    def __init__(self) -> None:
        self.captured: dict = {}
        self.messages = _FakeMessages(self.captured)
        self.api_key = "sk-test"


class _FakeCompletions:
    def __init__(self, sink: dict) -> None:
        self._sink = sink

    def create(self, **kwargs):
        self._sink.update(kwargs)
        return "openai-response"


class _FakeChat:
    def __init__(self, sink: dict) -> None:
        self.completions = _FakeCompletions(sink)


class _FakeOpenAI:
    def __init__(self) -> None:
        self.captured: dict = {}
        self.chat = _FakeChat(self.captured)


# ── anthropic ─────────────────────────────────────────────────────────────

def test_anthropic_wrapper_compresses_messages():
    raw = _FakeAnthropic()
    client = with_anthropic(raw)
    messages = [
        {"role": "user", "content": "list users"},
        {"role": "user", "content": _big_rows()},
    ]
    result = client.messages.create(model="claude-sonnet-4-5", messages=messages)

    assert result == "anthropic-response"
    sent = raw.captured["messages"]
    # The big JSON message was rewritten into a columnar block.
    assert "<<fb:columnar" in sent[-1]["content"]


def test_anthropic_wrapper_delegates_other_attrs():
    raw = _FakeAnthropic()
    client = with_anthropic(raw)
    assert client.api_key == "sk-test"
    assert client.messages.count_tokens() == 123


# ── openai ─────────────────────────────────────────────────────────────────

def test_openai_wrapper_compresses_messages():
    raw = _FakeOpenAI()
    client = with_openai(raw)
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": _big_rows()},
    ]
    result = client.chat.completions.create(model="gpt-4o", messages=messages)

    assert result == "openai-response"
    sent = raw.captured["messages"]
    assert "<<fb:columnar" in sent[-1]["content"]
    # System message (not last) is untouched.
    assert sent[0]["content"] == "be helpful"


def test_wrapper_passes_through_when_no_messages():
    raw = _FakeOpenAI()
    client = with_openai(raw)
    client.chat.completions.create(model="gpt-4o")  # no messages kwarg
    assert "messages" not in raw.captured
