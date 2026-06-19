"""Drop-in wrappers for the Anthropic and OpenAI SDK clients.

Each wrapper returns a transparent proxy: every attribute and method behaves
exactly like the underlying client, except the ``create`` call on the
chat/messages endpoint runs ``compress`` on the ``messages`` list first.

    from anthropic import Anthropic
    from foldback.integrations import with_anthropic

    client = with_anthropic(Anthropic())
    client.messages.create(model="claude-sonnet-4-5", messages=[...])  # compressed

    from openai import OpenAI
    from foldback.integrations import with_openai

    client = with_openai(OpenAI())
    client.chat.completions.create(model="gpt-4o", messages=[...])     # compressed

No network code lives here and the SDKs are not imported — you pass in an
already-constructed client, so these work with any SDK version.
"""

from __future__ import annotations

from .wrappers import with_anthropic, with_openai

__all__ = ["with_anthropic", "with_openai"]
