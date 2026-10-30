"""Stage 2: one score per candidate (sketch §7, namespace weights from §9).

S(u | q) = n(u) · v(u) · q(u) · f(u) · rel(u | q)

rel(u | q) = [ w_s·cos(q̂, s_u) + w_c·cos(q̂, c_u) + w_ℓ·bm25̃(q, u) + w_g·g(u)
               + w_y·sym(q, u) + w_p·path(q, u) ] / (w_s + w_c + w_ℓ + w_g + w_y + w_p)

q(u) = a + (1 − a)·Q(u) and f(u) = b + (1 − b)·F(u) are the usage and freshness priors
as multipliers in [a, 1] and [b, 1]. The sketch added them (w_Q·Q + w_F·F), which gave
every candidate the same ~0.175 whatever its relevance, so weak hits scored close to
strong ones and padded the budget; as multipliers they only modulate relevance.

n(u) is the namespace weight (session ×1.2). Cosines are clipped at 0 so an
anti-correlated view counts as no evidence rather than negative evidence, and a
missing view (no vector for the query's model) counts as 0.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from datetime import datetime

from pydantic import BaseModel

from codebrain.config import PagingParams, RetrievalParams
from codebrain.models import Namespace, Unit

RELEVANCE = ("semantic", "code", "lexical", "graph", "symbol", "path")
FEATURES = (*RELEVANCE, "quality", "freshness")


def freshness(unit: Unit, now: datetime, params: RetrievalParams, paging: PagingParams) -> float:
    """F(u) = exp(−λ Δt). Session units decay on an hours scale instead of days."""
    age_s = max((now - unit.updated_at).total_seconds(), 0.0)
    if unit.namespace is Namespace.SESSION:
        lam = math.log(2) / (paging.session_half_life_hours * 3600)
    else:
        lam = params.freshness_lambda / 86400
    return math.exp(-lam * age_s)


def path_match(unit: Unit, terms: Collection[str]) -> float:
    """1 if a query term names a directory on the unit's path, 0.5 if it names the
    file's stem (``webhooks`` for ``pkg/webhooks.py``), else 0."""
    if not unit.path or not terms:
        return 0.0
    *dirs, name = unit.path.lower().split("/")
    if any(d in terms for d in dirs):
        return 1.0
    return 0.5 if name.rsplit(".", 1)[0] in terms else 0.0


def namespace_weight(unit: Unit, paging: PagingParams) -> float:
    return paging.session_weight if unit.namespace is Namespace.SESSION else 1.0


def weights(params: RetrievalParams) -> dict[str, float]:
    """The relevance weights, normalized to sum to 1."""
    raw = {
        "semantic": params.w_semantic,
        "code": params.w_code,
        "lexical": params.w_lexical,
        "graph": params.w_graph,
        "symbol": params.w_symbol,
        "path": params.w_path,
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()} if total > 0 else raw


class ScoreBreakdown(BaseModel):
    """Every input to S(u | q) and the weighted share each one contributed."""

    semantic: float
    code: float
    lexical: float
    graph: float
    symbol: float = 0.0
    path: float = 0.0
    quality: float
    freshness: float
    validity: float
    namespace_weight: float
    contributions: dict[str, float]
    relevance: float
    """rel(u | q): the normalized weighted sum of the relevance features, in [0, 1]."""
    prior: float = 1.0
    """q(u) · f(u), the usage and freshness multiplier."""
    score: float

    def features(self) -> dict[str, float]:
        """The flat feature vector logged with each retrieval, for learning the weights."""
        out = {name: getattr(self, name) for name in FEATURES}
        out["validity"] = self.validity
        out["namespace_weight"] = self.namespace_weight
        return out


def score(
    unit: Unit,
    *,
    semantic: float | None,
    code: float | None,
    lexical: float,
    graph: float,
    now: datetime,
    params: RetrievalParams,
    paging: PagingParams,
    symbol: float = 0.0,
    path: float = 0.0,
    boost: float = 0.0,
) -> ScoreBreakdown:
    """Score one candidate. ``lexical``, ``graph``, ``symbol`` and ``path`` are in [0, 1].

    ``boost`` is added to the relevance (a reranker's pick)."""
    values = {
        "semantic": max(semantic or 0.0, 0.0),
        "code": max(code or 0.0, 0.0),
        "lexical": lexical,
        "graph": graph,
        "symbol": symbol,
        "path": path,
    }
    contributions = {name: w * values[name] for name, w in weights(params).items()}
    relevance = sum(contributions.values()) + boost
    q, f = unit.quality, freshness(unit, now, params, paging)
    prior = (params.quality_floor + (1 - params.quality_floor) * q) * (
        params.freshness_floor + (1 - params.freshness_floor) * f
    )
    ns = namespace_weight(unit, paging)
    return ScoreBreakdown(
        **values,
        quality=q,
        freshness=f,
        validity=unit.validity,
        namespace_weight=ns,
        contributions=contributions,
        relevance=relevance,
        prior=prior,
        score=ns * unit.validity * prior * relevance,
    )
