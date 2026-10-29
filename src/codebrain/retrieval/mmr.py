"""Stage 3: diversity with maximal marginal relevance (sketch §7).

u* = argmax_{u ∈ C \\ R} [ μ·S(u | q) − (1 − μ)·max_{v ∈ R} sim(u, v) ]
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Pick:
    unit_id: str
    score: float
    """S(u | q)."""
    redundancy: float
    """max sim to the units picked before it (0 for the first)."""
    value: float
    """The MMR objective at the moment it was picked."""


def mmr(
    scores: Mapping[str, float],
    sim: Callable[[str, str], float],
    *,
    mu: float = 0.7,
    limit: int | None = None,
) -> list[Pick]:
    """Order ``scores`` by greedy MMR. Ties go to the higher score, then the id."""
    remaining = sorted(scores, key=lambda u: (-scores[u], u))
    redundancy = dict.fromkeys(remaining, 0.0)
    picks: list[Pick] = []
    while remaining and (limit is None or len(picks) < limit):
        best = max(
            remaining,
            key=lambda u: (mu * scores[u] - (1 - mu) * redundancy[u], scores[u]),
        )
        value = mu * scores[best] - (1 - mu) * redundancy[best]
        picks.append(Pick(best, scores[best], redundancy[best], value))
        remaining.remove(best)
        for u in remaining:
            redundancy[u] = max(redundancy[u], sim(u, best))
    return picks
