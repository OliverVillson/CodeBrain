"""Embedding models for the summary view s_u and the code view c_u (sketch §4).

Models are undecided, so ``Embedder`` is a small protocol and ``make_embedder``
builds one from a spec string in ``EmbeddingParams``:

* ``hashing`` / ``hashing:<dim>``: feature hashing of identifier parts and word
  bigrams. No model download; lexical rather than semantic, so it is a baseline
  and the test default, not a production choice.
* ``sentence-transformers:<model>``: any local SentenceTransformer model
  (``pip install sentence-transformers``).
* anything added with ``register_embedder``.

The unchosen placeholders (``unset-*``) fall back to ``hashing`` with a warning so
ingest stays runnable. Every vector is L2-normalized, so cosine is a dot product.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from codebrain.ingest.lexical import split_identifier

log = logging.getLogger(__name__)


class Embedder(Protocol):
    model: str
    """Stored on each vector, so the index can be rebuilt when the model changes."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else list(vec)


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


class HashingEmbedder:
    """Signed feature hashing of identifier parts and their bigrams."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.model = f"hashing-v1-{dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = [p for w in _WORD.findall(text) for p in split_identifier(w)]
        feats = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        for feat in feats:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return normalize([math.copysign(math.log1p(abs(x)), x) for x in vec])


class SentenceTransformerEmbedder:
    def __init__(self, name: str, batch_size: int = 64) -> None:
        try:
            st = importlib.import_module("sentence_transformers")
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers embedders need: pip install sentence-transformers"
            ) from e
        self._model: Any = st.SentenceTransformer(name)
        self.model = f"sentence-transformers:{name}"
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = self._model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)
        return [list(map(float, row)) for row in out]


EmbedderFactory = Callable[[str], Embedder]
_REGISTRY: dict[str, EmbedderFactory] = {}


def register_embedder(prefix: str, factory: EmbedderFactory) -> None:
    """Make ``<prefix>:<arg>`` a valid embedding model spec."""
    _REGISTRY[prefix] = factory


def make_embedder(spec: str, dim: int = 1024, batch_size: int = 64) -> Embedder:
    prefix, _, arg = spec.partition(":")
    if prefix.startswith("unset"):
        log.warning("embedding model %r is not chosen yet; using the hashing baseline", spec)
        return HashingEmbedder(dim)
    if prefix == "hashing":
        return HashingEmbedder(int(arg) if arg else dim)
    if prefix == "sentence-transformers":
        return SentenceTransformerEmbedder(arg, batch_size=batch_size)
    if prefix in _REGISTRY:
        return _REGISTRY[prefix](arg)
    raise ValueError(f"unknown embedding model {spec!r}")
