# Changelog

All notable changes to FoldBack are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.1] — 2026-06-20

### Changed
- Author metadata set to Sudarshan Chaudhari (shows on the PyPI page).
- README: added Author, License, and Acknowledgments (credit to
  `chopratejas/headroom`) sections.
- NOTICE copyright updated.

No code or behavior changes from 0.1.0.

## [0.1.0] — 2026-06-20

First production-ready release.

### Added
- `compress(messages, model=...)` — live-zone-only, content-preserving
  compression returning a `CompressResult` (`messages`, `tokens_saved`,
  `ratio`, `transforms`).
- `CompressConfig` — `model`, `min_savings` token-savings gate, per-transform
  `enabled` toggles.
- Reversible JSON columnar compactor with a proven inverse (`restore_columnar`)
  and Hypothesis property tests guaranteeing exact round-trips.
- Normalizing log compactor (ANSI strip + consecutive-line run-length dedup)
  and text compactor (trailing whitespace, blank-line runs).
- Optional exact token counting via `tiktoken` (`foldback[exact]`); zero-dep
  char/4 heuristic by default.
- Drop-in SDK wrappers `with_anthropic()` / `with_openai()`.
- Full quality gate: ruff, mypy `--strict`, pytest with coverage, Python
  3.10–3.13, packaged wheel + sdist (`twine check` clean).

### Correctness
- Token-based savings gate: a transform is applied only when it reduces
  tokens (not merely bytes).
- Cache hot zone (everything before the last `cache_control` breakpoint) is
  forwarded as the identical objects received — never re-serialized.

### Fixed (vs. the pre-release prototype)
- JSON compaction was not reversible: string `"1"` and integer `1` rendered
  identically, and the absent-key sentinel collided with real values. The
  columnar format now keeps every cell as a typed JSON token and only
  compacts uniform-schema arrays, so the round-trip is exact.
