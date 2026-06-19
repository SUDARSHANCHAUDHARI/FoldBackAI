"""Transparent compression proxies for LLM SDK clients.

The proxies wrap only the ``create`` callable at the end of the endpoint
chain (``client.messages.create`` for Anthropic, ``client.chat.completions
.create`` for OpenAI). Everything else is delegated to the real object via
``__getattr__``, so the proxy is indistinguishable from the wrapped client
for any other use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..compress import CompressConfig, compress


def _compress_kwargs(kwargs: dict[str, Any], config: CompressConfig | None) -> None:
    """Compress ``kwargs['messages']`` in place, if present and a list."""
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        model = kwargs.get("model")
        cfg = config if config is not None else CompressConfig(model=model)
        kwargs["messages"] = compress(messages, config=cfg).messages


class _CreateProxy:
    """Wraps a ``create`` callable, compressing messages before each call."""

    def __init__(self, create: Callable[..., Any], config: CompressConfig | None) -> None:
        self._create = create
        self._config = config

    def create(self, *args: Any, **kwargs: Any) -> Any:
        _compress_kwargs(kwargs, self._config)
        return self._create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegate anything else (e.g. .stream, .with_raw_response) to the
        # original endpoint object that owns the create callable.
        return getattr(self._create.__self__, name)  # type: ignore[attr-defined]


class _AnthropicProxy:
    def __init__(self, client: Any, config: CompressConfig | None) -> None:
        self._client = client
        self._config = config

    @property
    def messages(self) -> _CreateProxy:
        return _CreateProxy(self._client.messages.create, self._config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _OpenAINamespace:
    def __init__(self, completions: _CreateProxy) -> None:
        self.completions = completions


class _OpenAIProxy:
    def __init__(self, client: Any, config: CompressConfig | None) -> None:
        self._client = client
        self._config = config

    @property
    def chat(self) -> _OpenAINamespace:
        proxy = _CreateProxy(self._client.chat.completions.create, self._config)
        return _OpenAINamespace(proxy)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def with_anthropic(client: Any, *, config: CompressConfig | None = None) -> _AnthropicProxy:
    """Wrap an Anthropic client so ``messages.create`` compresses messages.

    The ``system`` prompt is a separate parameter on the Anthropic API and is
    never touched — which is exactly what keeps the prompt cache warm.
    """
    return _AnthropicProxy(client, config)


def with_openai(client: Any, *, config: CompressConfig | None = None) -> _OpenAIProxy:
    """Wrap an OpenAI client so ``chat.completions.create`` compresses messages.

    With no ``cache_control`` markers (OpenAI has none in the message list),
    only the final message is treated as live — the safe default.
    """
    return _OpenAIProxy(client, config)
