# FoldBack

**Lossless context compression for LLM agents.** Fold it down, fold it back.

FoldBack shrinks what your agent sends to the model — JSON tool outputs, logs,
search results — using only **lossless, data-preserving** transforms. No row
is dropped. No value is discarded. The provider prompt cache keeps hitting.

```python
from foldback import compress

result = compress(messages)
result.messages       # same format, fewer tokens — send these to the model
result.tokens_saved   # estimated tokens saved
result.ratio          # 0.4 == 60% saved; 1.0 == nothing changed
```

## Why it exists

Most "context compression" tools (e.g. Headroom) made two expensive mistakes:

1. **They compressed conversation history**, dropping old messages — which
   busts the provider prompt cache on every call. On Anthropic that's a 90%
   discount thrown away.
2. **They dropped data and hoped the model would ask for it back** via a
   retrieval tool. If it doesn't realize data is missing, you get a
   confidently wrong answer with no error.

FoldBack refuses both:

- **Passthrough is sacred.** Everything before the last `cache_control`
  breakpoint is forwarded as the *same objects* — never copied, never
  re-serialized — so caches keep hitting.
- **Only the live zone is touched** (the latest user message / tool result).
- **Every transform is lossless.** A JSON array becomes a columnar table; a
  log run-length-collapses identical lines with an explicit `(xN)` count.
  Representation changes, data does not.
- **Self-gating.** If a transform doesn't shrink the content, the original is
  returned unchanged.

## What it does today

| Content | Transform | Lossless? |
|---------|-----------|-----------|
| JSON array of objects | columnar table (keys written once) | yes — full data reconstructable |
| Logs | strip ANSI, collapse duplicate lines to `(xN)` | yes — counts preserved |
| Plain text | trim trailing whitespace, collapse blank-line runs | yes — words/punctuation untouched |

No ML model, no downloads, **zero dependencies**, Python 3.10+.

## Install & test

```bash
pip install -e ".[dev]"
pytest -q
python examples/demo.py
```

## Deliberately NOT built

The proxy, SSE streaming parser, Bedrock/Vertex signing, message scoring /
relevance, a HuggingFace compression model, lossy row-dropping with retrieval.
Those are where the competition bled. FoldBack is a library you call before
your own SDK call — so it can never corrupt the wire.

## Roadmap

- [ ] Diff / patch compaction
- [ ] Optional exact tokenizer (tiktoken / anthropic) behind `tokens.estimate`
- [ ] Rust core for the JSON columnar path (only if profiling demands it)

Apache 2.0.
