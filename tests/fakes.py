"""Test doubles for retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from codebrain.retrieval.search import lexical_terms


def _stem(term: str) -> str:
    for suffix in ("ing", "er", "ed", "s"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            return term[: -len(suffix)]
    return term


class HashEmbedder:
    """Bag of stemmed terms hashed into ``dim`` buckets. Deterministic, no model needed."""

    def __init__(self, model: str = "hash-test", dim: int = 256) -> None:
        self.model = model
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for t in lexical_terms(text):
                h = int(hashlib.md5(_stem(t).encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            out.append(vec)
        return out
