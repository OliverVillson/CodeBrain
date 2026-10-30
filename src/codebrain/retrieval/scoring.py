"""Stage 2: one score per candidate (sketch §7, namespace weights from §9).

S(u | q) = n(u) · v(u) · [ w_s·cos(q̂, s_u) + w_c·cos(q̂, c_u) + w_ℓ·bm25̃(q, u)
                         + w_g·g(u) + w_Q·Q(u) + w_F·F(u) ]

n(u) is the namespace weight (session ×1.2). Cosines are clipped at 0 so an
anti-correlated view counts as no evidence rather than negative evidence, and a
missing view (no vector for the query's model) counts as 0.
"""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel

from codebrain.config import PagingParams, RetrievalParams
from codebrain.models import Namespace, Unit

FEATURES = ("semantic", "code", "lexical", "graph", "quality", "freshness")


def freshness(unit: Unit, now: datetime, params: RetrievalParams, paging: PagingParams) -> float:
    """F(u) = exp(−λ Δt). Session units decay on an hours scale instead of days."""
    age_s = max((now - unit.updated_at).total_seconds(), 0.0)
    if unit.namespace is Namespace.SESSION:
        lam = math.log(2) / (paging.session_half_life_hours * 3600)
    else:
        lam = params.freshness_lambda / 86400
    return math.exp(-lam * age_s)


def namespace_weight(unit: Unit, paging: PagingParams) -> float:
    return paging.session_weight if unit.namespace is Namespace.SESSION else 1.0


def weights(params: RetrievalParams) -> dict[str, float]:
    return {
        "semantic": params.w_semantic,
        "code": params.w_code,
        "lexical": params.w_lexical,
        "graph": params.w_graph,
        "quality": params.w_quality,
        "freshness": params.w_freshness,
    }


class ScoreBreakdown(BaseModel):
    """Every input to S(u | q) and the weighted share each one contributed."""

    semantic: float
    code: float
    lexical: float
    graph: float
    quality: float
    freshness: float
    validity: float
    namespace_weight: float
    contributions: dict[str, float]
    relevance: float
    """The bracketed weighted sum, before validity and namespace weight."""
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
) -> ScoreBreakdown:
    """Score one candidate. ``lexical`` and ``graph`` are already normalized to [0, 1]."""
    values = {
        "semantic": max(semantic or 0.0, 0.0),
        "code": max(code or 0.0, 0.0),
        "lexical": lexical,
        "graph": graph,
        "quality": unit.quality,
        "freshness": freshness(unit, now, params, paging),
    }
    contributions = {name: w * values[name] for name, w in weights(params).items()}
    relevance = sum(contributions.values())
    ns = namespace_weight(unit, paging)
    return ScoreBreakdown(
        **values,
        validity=unit.validity,
        namespace_weight=ns,
        contributions=contributions,
        relevance=relevance,
        score=ns * unit.validity * relevance,
    )
