"""Token counting for the τ bounds (sketch §3).

The bounds only need to be roughly right, so the default counter approximates a
BPE tokenizer without loading one. Swap in an exact counter for the embedding
model once it is chosen.
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


def approx_tokens(text: str) -> int:
    """Count words, digit runs and punctuation, plus one more per 8 letters of a long word."""
    n = 0
    for m in _TOKEN_RE.finditer(text):
        n += 1 + (len(m.group()) - 1) // 8
    return n
