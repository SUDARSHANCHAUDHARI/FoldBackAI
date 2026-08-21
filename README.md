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

## Table of Contents

- [Overview](#overview)
- [Guarantees, stated honestly](#guarantees-stated-honestly)
- [Measured savings](#measured-savings)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Deliberately NOT built](#deliberately-not-built)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [About](#about)

## Overview

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

## Installation

The PyPI package is **`foldback-ai`**; the import name is **`foldback`**.

```bash
pip install foldback-ai                 # zero dependencies
pip install "foldback-ai[exact]"        # + tiktoken for exact token counts
pip install "foldback-ai[anthropic]"    # + Anthropic SDK for the wrapper
pip install "foldback-ai[openai]"       # + OpenAI SDK for the wrapper
```

```python
from foldback import compress           # import name stays `foldback`
```

## Usage

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

## Development

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

## Acknowledgments

FoldBack's design was informed by studying [chopratejas/headroom](https://github.com/chopratejas/headroom)
— a more ambitious context-compression project. FoldBack deliberately takes a
narrower, library-only path (no proxy, lossless-first, prompt-cache-preserving)
to avoid the cache-busting and lossy-retrieval pitfalls that project documented
in its own realignment notes. Credit to its authors for mapping the problem space.

## License

Apache 2.0 © Sudarshan Chaudhari / SudarshanTechLabs. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

## About

I'm Sudarshan Chaudhari, a Senior Quality Engineer, Test Automation specialist, and AI systems builder based in Bangkok, Thailand.

I have 13+ years of experience in software quality engineering, working across SaaS, fintech, gaming, web, mobile, cloud, and digital signage platforms. My background combines hands-on test automation with QA leadership, test strategy, CI/CD, release quality, production investigation, and cross-platform validation.

Alongside my professional QA career, I run [SudarshanTechLabs](https://sudarshantechlabs.com/), my independent engineering and product lab where I design, build, test, and ship software across Android, web, AI, cybersecurity, developer tooling, and cross-platform applications.

### What I work on

- ⚙️ **Quality Engineering & Test Automation** — Playwright, Selenium, Cypress, Appium, API testing, automation frameworks, end-to-end testing, CI/CD, release gates, GitHub Actions, risk-based testing, and production validation
- 🤖 **AI Systems & Automation** — AI agents, multi-agent orchestration, MCP servers, AI-assisted QA, prompt tooling, developer workflows, automation systems, and Claude Code plugins
- 📱 **Mobile & Cross-Platform Applications** — Android applications built with Kotlin and Jetpack Compose, Google Play releases, automated build and publishing pipelines, and cross-platform development spanning iOS, web, Windows, and macOS
- 🌐 **Web Applications & Platforms** — Full-stack applications using Next.js, TypeScript, Firebase, Cloudflare, REST APIs, and modern web infrastructure
- 🛠️ **Developer Tooling & CLI Engineering** — Rust, Python, TypeScript, CLI utilities, multi-repository tooling, build automation, release tooling, and engineering productivity systems
- 🛡️ **Cybersecurity & Observability** — Threat detection, log analysis, security auditing, vulnerability assessment, monitoring, and security-focused developer tools
- 📺 **Digital Signage & Device Platforms** — Content validation, playback testing, device compatibility, production investigation, monitoring, and QA across diverse hardware and operating-system environments

My work sits at the intersection of quality engineering, automation, AI, and software development. I approach products with a QA mindset from the beginning: understanding failure modes, designing for testability, automating repetitive work, and building release confidence into the engineering process.

Through SudarshanTechLabs, I also build products and tools from idea to production, covering architecture, development, testing, CI/CD, release automation, monitoring, and ongoing maintenance.

🌐 [sudarshantechlabs.com](https://sudarshantechlabs.com/) · 💼 [LinkedIn](https://linkedin.com/in/sudarshan-chaudhari) · 🐙 [GitHub](https://github.com/SUDARSHANCHAUDHARI) · ✉️ [sunny.sudarshan@gmail.com](mailto:sunny.sudarshan@gmail.com)
