"""Reciprocal rank fusion across the stage 1 indexes (sketch §7)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def rrf(rankings: Mapping[str, Sequence[str]], k: int = 60) -> dict[str, float]:
    """RRF(u) = Σ_i 1 / (k + rank_i(u)), ranks from 1. Sorted best first, ties by id."""
    scores: dict[str, float] = {}
    for ids in rankings.values():
        for rank, uid in enumerate(ids, start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])))


def ranks_of(rankings: Mapping[str, Sequence[str]], unit_id: str) -> dict[str, int]:
    """1-based rank of ``unit_id`` in each index that returned it."""
    return {name: ids.index(unit_id) + 1 for name, ids in rankings.items() if unit_id in ids}
