# FoldBack

**Context compression for LLM agents.** Fold it down, fold it back.

FoldBack shrinks what your agent sends to the model — JSON tool outputs, logs,
search results — using content-preserving transforms. No row is dropped, no
value is discarded, and the provider prompt cache keeps hitting.

```python
from foldback import compress

result = compress(messages, model="gpt-4o")
result.messages       # same format, fewer tokens — send these to the model
result.tokens_saved   # tokens saved
result.ratio          # 0.45 == 55% saved; 1.0 == nothing changed
result.transforms     # e.g. ["json:columnar"]
```

## Why it exists

Most "context compression" tools made two expensive mistakes:

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
- **Token-gated.** A transform is applied only when it actually reduces
  tokens. Otherwise the original is returned unchanged.

## Guarantees, stated honestly

Two transform categories with different promises:

| Content | Transform | Guarantee |
|---------|-----------|-----------|
| JSON array of uniform objects | columnar (keys written once, rows as JSON arrays) | **Reversible** — exact round-trip, proven by property tests. `restore_columnar()` reconstructs the original. |
| Logs | strip ANSI, run-length-collapse consecutive identical lines to `(xN)` | **Normalizing** — no textual content lost; removes non-semantic bytes. Not byte-reversible. |
| Plain text | trim trailing whitespace, collapse blank-line runs | **Normalizing** — words and punctuation untouched. |

The columnar transform only fires on **uniform-schema** arrays, so each row
maps back to its keys unambiguously and `"1"` (string) never collides with `1`
(number). Mixed-schema arrays are left untouched rather than compacted lossily.

```python
from foldback import compress, restore_columnar
# round-trip proof
compressed = compress(messages).messages
# any columnar block is exactly restorable:
#   json.loads(restore_columnar(block)) == original_rows
```

## Measured savings

Reproduce with `python benchmarks/run.py --model gpt-4o` (exact gpt-4o tokens):

| Workload | Before | After | Saved |
|----------|-------:|------:|------:|
| API response (100 rows) | 2,803 | 1,421 | **49%** |
| Build log (200 lines) | 2,729 | 499 | **82%** |
| Code search (50 hits) | 1,892 | 1,159 | **39%** |

No marketing numbers — these come straight from the benchmark script.

## Install

The PyPI package is **`foldbackai`**; the import name is **`foldback`**.

```bash
pip install foldbackai                 # zero dependencies
pip install "foldbackai[exact]"        # + tiktoken for exact token counts
pip install "foldbackai[anthropic]"    # + Anthropic SDK for the wrapper
pip install "foldbackai[openai]"       # + OpenAI SDK for the wrapper
```

```python
from foldback import compress           # import name stays `foldback`
```

## Use it

**Inline:**
```python
from foldback import compress, CompressConfig

result = compress(messages, model="claude-sonnet-4-5")
# or with options:
cfg = CompressConfig(model="gpt-4o", min_savings=0.2)  # require >=20% win
result = compress(messages, config=cfg)
```

**Drop-in SDK wrappers** (system prompt / tool defs stay frozen → cache-safe):
```python
from anthropic import Anthropic
from foldback.integrations import with_anthropic

client = with_anthropic(Anthropic())
client.messages.create(model="claude-sonnet-4-5", messages=[...])  # auto-compressed

from openai import OpenAI
from foldback.integrations import with_openai

client = with_openai(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=[...])     # auto-compressed
```

## Develop

```bash
pip install -e ".[dev,exact]"
pytest                       # tests + coverage
ruff check foldback tests    # lint
mypy foldback                # strict type-check
python benchmarks/run.py     # savings table
python examples/demo.py
```

## Deliberately NOT built

A network proxy, SSE streaming parser, Bedrock/Vertex signing, message
scoring / relevance, a HuggingFace compression model, lossy row-dropping with
retrieval. FoldBack is a library you call before your own SDK call — so it can
never corrupt the wire.

## Roadmap

- [ ] Diff / patch compaction
- [ ] CSV / Markdown-table input detection
- [ ] Rust core for the columnar path (only if profiling demands it)

## Author

Built and maintained by **Sudarshan Chaudhari**
([@SUDARSHANCHAUDHARI](https://github.com/SUDARSHANCHAUDHARI),
sunny.sudarshan@gmail.com) — SudarshanTechLabs, Bangkok.

If FoldBack saves you tokens, a ⭐ on the
[repo](https://github.com/SUDARSHANCHAUDHARI/FoldBackAI) is appreciated.

<sub>Developed with Claude Code (Opus 4.8).</sub>

## Acknowledgments

FoldBack's design was informed by studying [chopratejas/headroom](https://github.com/chopratejas/headroom)
— a more ambitious context-compression project. FoldBack deliberately takes a
narrower, library-only path (no proxy, lossless-first, prompt-cache-preserving)
to avoid the cache-busting and lossy-retrieval pitfalls that project documented
in its own realignment notes. Credit to its authors for mapping the problem space.

## License

Apache 2.0 © Sudarshan Chaudhari / SudarshanTechLabs. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
