"""Retrieval metrics (plan §3) with bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    relevant = {u for u, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr(ranked: Sequence[str], rel: Mapping[str, int]) -> float:
    for i, u in enumerate(ranked, 1):
        if rel.get(u, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    def dcg(grades: Sequence[int]) -> float:
        return sum(((1 << g) - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    ideal = dcg(sorted(rel.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([rel.get(u, 0) for u in ranked[:k]]) / ideal


def budget_recall(
    ranked: Sequence[str], rel: Mapping[str, int], tokens: Callable[[str], int], budget: int
) -> float:
    """Graded share of the relevant set that fits when packing full units in rank order."""
    total = sum(rel.values())
    if total == 0:
        return 0.0
    used, got = 0, 0
    for u in ranked:
        t = tokens(u)
        if used + t > budget:
            continue
        used += t
        got += rel.get(u, 0)
    return got / total


def bootstrap_ci(
    values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(rng.choices(values, k=m)) / m for _ in range(n))
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]
