"""Run: python examples/demo.py

Shows FoldBack compressing a realistic JSON tool result and a noisy log,
while leaving the cache hot zone untouched.
"""

from __future__ import annotations

import json

from foldback import compress

# A system prompt the agent reuses every turn -> mark it cache-frozen.
system = {
    "role": "system",
    "content": "You are a helpful agent.",
    "cache_control": {"type": "ephemeral"},
}

# A chunky JSON tool result (50 user rows) in the live zone.
users = [
    {"id": i, "name": f"user_{i}", "email": f"u{i}@example.com", "active": True}
    for i in range(50)
]

# A noisy log with repeats + ANSI color.
log = "boot\n" + "\x1b[33mWARN retrying connection\x1b[0m\n" * 12 + "ready"

messages = [
    system,
    {"role": "user", "content": "list active users and check the logs"},
    {"role": "tool", "content": json.dumps(users)},
    {"role": "tool", "content": log},
]

result = compress(messages)

print("transforms applied :", result.transforms)
print("original tokens    :", result.original_tokens)
print("compressed tokens  :", result.compressed_tokens)
print("tokens saved       :", result.tokens_saved)
print(f"ratio              : {result.ratio:.2f}  ({(1 - result.ratio) * 100:.0f}% saved)")
print("cache hot zone kept:", result.messages[0] is system)
