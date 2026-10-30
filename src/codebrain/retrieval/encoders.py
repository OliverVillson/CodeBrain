"""Query encoders: what the broker needs to embed a query (sketch §4).

Embedding models are undecided (sketch §10), so recall takes one ``Embedder`` per
dense view. Each must be the same model ingest used for that view; its ``model``
id selects the matching rows in ``unit_vectors``. A view with no encoder is
skipped, so lexical-only recall works before any model is picked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from codebrain.models import VectorView

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    @property
    def model(self) -> str:
        """Model id, as stored in ``unit_vectors.model``."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


Encoders = Mapping[VectorView, Embedder]


def encode_query(encoders: Encoders, query: str) -> dict[VectorView, tuple[str, Vector]]:
    """Embed ``query`` once per available view: ``{view: (model, unit-length vector)}``."""
    out: dict[VectorView, tuple[str, Vector]] = {}
    for view, enc in encoders.items():
        [vec] = enc.embed([query])
        out[view] = (enc.model, normalize(np.asarray(vec, dtype=np.float32)))
    return out


def normalize(v: Vector) -> Vector:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: Vector | None, b: Vector | None) -> float | None:
    """Cosine similarity, or ``None`` when either side is missing or the dims differ."""
    if a is None or b is None or a.shape != b.shape:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
