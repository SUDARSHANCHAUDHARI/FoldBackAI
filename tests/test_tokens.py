"""Tests for token counting — heuristic and exact backends."""

from __future__ import annotations

from foldback import tokens


def test_empty_is_zero():
    assert tokens.estimate("") == 0


def test_heuristic_scales_with_length():
    assert tokens.estimate("a" * 40) == 10  # 40 / 4


def test_heuristic_minimum_one():
    assert tokens.estimate("x") == 1


def test_unknown_model_falls_back_to_heuristic_or_default():
    # Whether or not tiktoken is installed, an unknown model must not raise.
    n = tokens.estimate("hello world", model="some-nonexistent-model-xyz")
    assert n >= 1


def test_exact_backend_when_available():
    import pytest

    pytest.importorskip("tiktoken")
    # A known OpenAI model should yield an exact count > 0 and typically
    # differ from the crude char/4 heuristic on natural text.
    exact = tokens.estimate("The quick brown fox jumps.", model="gpt-4o")
    assert exact > 0
