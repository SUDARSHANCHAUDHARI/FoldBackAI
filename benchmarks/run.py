"""Reproducible savings benchmark.

Run: python benchmarks/run.py [--model gpt-4o]

Generates representative agent tool outputs, compresses each, and prints a
token-savings table. With --model and tiktoken installed, counts are exact.
The point is honesty: the numbers in the README come from running this.
"""

from __future__ import annotations

import argparse
import json

from foldback import compress
from foldback.compress import CompressConfig


def _api_rows(n: int = 100) -> str:
    return json.dumps(
        [
            {"id": i, "name": f"user_{i}", "email": f"u{i}@example.com", "active": i % 3 == 0}
            for i in range(n)
        ]
    )


def _noisy_log(n: int = 200) -> str:
    lines = ["\x1b[32mINFO server started\x1b[0m"]
    for i in range(n):
        if i % 7 == 0:
            lines.append(f"\x1b[31mERROR conn reset (attempt {i})\x1b[0m")
        else:
            lines.append("\x1b[33mWARN retrying connection\x1b[0m")
    return "\n".join(lines)


def _search_hits(n: int = 50) -> str:
    return json.dumps(
        [
            {"path": f"src/module_{i}.py", "line": i * 3, "score": round(1 / (i + 1), 4),
             "snippet": "def handler(request): return process(request)"}
            for i in range(n)
        ]
    )


WORKLOADS = {
    "API response (100 rows)": _api_rows(),
    "Build log (200 lines)": _noisy_log(),
    "Code search (50 hits)": _search_hits(),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="model id for exact tokenization")
    args = parser.parse_args()
    cfg = CompressConfig(model=args.model)

    print(f"{'Workload':<26} {'Before':>8} {'After':>8} {'Saved':>7}  Transform")
    print("-" * 70)
    for name, payload in WORKLOADS.items():
        messages = [
            {"role": "user", "content": "process this"},
            {"role": "tool", "content": payload},
        ]
        r = compress(messages, config=cfg)
        pct = f"{(1 - r.ratio) * 100:.0f}%"
        transform = r.transforms[0] if r.transforms else "(none)"
        print(
            f"{name:<26} {r.original_tokens:>8} {r.compressed_tokens:>8} {pct:>7}  {transform}"
        )


if __name__ == "__main__":
    main()
