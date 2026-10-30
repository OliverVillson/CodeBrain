"""Retrieval math that needs no database: RRF, PageRank, scoring, MMR, packing."""

from datetime import UTC, datetime, timedelta

import pytest

from codebrain.config import Settings
from codebrain.models import Edge, EdgeType, Namespace, Resolution, Span, Unit, UnitKind
from codebrain.retrieval import pack
from codebrain.retrieval.fusion import ranks_of, rrf
from codebrain.retrieval.graph import personalized_pagerank
from codebrain.retrieval.mmr import mmr
from codebrain.retrieval.scoring import freshness, score
from codebrain.retrieval.search import lexical_query, lexical_terms

NOW = datetime(2026, 10, 29, tzinfo=UTC)
S = Settings()


def _unit(**kw: object) -> Unit:
    base: dict[str, object] = dict(
        kind=UnitKind.FUNCTION, path="a.py", text="x" * 400, updated_at=NOW, tokens=100
    )
    base.update(kw)
    return Unit.model_validate(base)


def test_rrf_sums_reciprocal_ranks() -> None:
    fused = rrf({"a": ["x", "y"], "b": ["y", "z"]}, k=60)
    assert list(fused) == ["y", "x", "z"]
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert ranks_of({"a": ["x", "y"], "b": ["y"]}, "y") == {"a": 2, "b": 1}


def test_pagerank_decays_with_hops_and_walks_both_ways() -> None:
    # a -> b -> c: c is only reachable from the seed a by following edges forwards,
    # and d -> a only backwards. Both must get mass.
    edges = [
        Edge(src_id="a", dst_id="b", type=EdgeType.CALLS),
        Edge(src_id="b", dst_id="c", type=EdgeType.CALLS),
        Edge(src_id="d", dst_id="a", type=EdgeType.CALLS),
    ]
    r = personalized_pagerank({"a": 1.0}, edges, beta=0.5)
    assert r["a"] > r["b"] > r["c"] > 0
    assert r["d"] > 0
    assert sum(r.values()) == pytest.approx(1.0)


def test_pagerank_without_edges_is_the_seed_distribution() -> None:
    r = personalized_pagerank({"a": 3.0, "b": 1.0}, [])
    assert r == pytest.approx({"a": 0.75, "b": 0.25})


def test_score_matches_sketch_formula() -> None:
    u = _unit(retrieved_count=3, used_count=3)  # Q = 0.8
    b = score(
        u,
        semantic=0.76,
        code=0.72,
        lexical=0.55,
        graph=0.60,
        now=NOW,
        params=S.retrieval,
        paging=S.paging,
    )
    expected = 0.35 * 0.76 + 0.15 * 0.72 + 0.15 * 0.55 + 0.10 * 0.60 + 0.15 * 0.8 + 0.10 * 1.0
    assert b.score == pytest.approx(expected)
    assert sum(b.contributions.values()) == pytest.approx(b.relevance)


def test_validity_and_namespace_scale_the_score() -> None:
    kw = dict(semantic=0.8, code=0.5, lexical=0.5, graph=0.0, now=NOW, params=S.retrieval)
    fresh = score(_unit(), paging=S.paging, **kw)  # type: ignore[arg-type]
    stale = score(_unit(validity=0.3), paging=S.paging, **kw)  # type: ignore[arg-type]
    session = score(
        _unit(kind=UnitKind.TURN, path=None, namespace=Namespace.SESSION, session_id="s"),
        paging=S.paging,
        **kw,  # type: ignore[arg-type]
    )
    assert stale.score == pytest.approx(0.3 * fresh.score)
    assert session.score == pytest.approx(1.2 * fresh.score)


def test_freshness_half_lives() -> None:
    month = _unit(updated_at=NOW - timedelta(days=30))
    assert freshness(month, NOW, S.retrieval, S.paging) == pytest.approx(0.5)
    turn = _unit(
        kind=UnitKind.TURN,
        path=None,
        namespace=Namespace.SESSION,
        session_id="s",
        updated_at=NOW - timedelta(hours=2),
    )
    assert freshness(turn, NOW, S.retrieval, S.paging) == pytest.approx(0.5)


def test_mmr_demotes_near_duplicates() -> None:
    scores = {"a": 0.8, "a2": 0.79, "b": 0.6}
    sims = {frozenset({"a", "a2"}): 0.99}
    picks = mmr(scores, lambda x, y: sims.get(frozenset({x, y}), 0.0), mu=0.7)
    assert [p.unit_id for p in picks] == ["a", "b", "a2"]
    assert picks[2].redundancy == pytest.approx(0.99)


def test_pack_respects_budget_and_upgrades_best_first() -> None:
    big = _unit(id="big", signature="def big()", summary="Does the big thing.", tokens=400)
    small = _unit(id="small", signature="def small()", summary="Small.", tokens=50)
    cands = {"big": pack.options(big, 0.9), "small": pack.options(small, 0.5)}

    roomy = pack.pack(cands, 10_000)
    assert {u: o.resolution for u, o in roomy.items()} == {
        "big": Resolution.FULL,
        "small": Resolution.FULL,
    }

    tight = pack.pack(cands, 120)
    assert sum(o.tokens for o in tight.values()) <= 120
    assert tight["small"].resolution is Resolution.FULL
    assert tight["big"].resolution is not Resolution.FULL

    assert pack.pack(cands, 0) == {}


def test_pack_skips_missing_resolutions() -> None:
    u = _unit(signature=None, summary=None)
    assert [o.resolution for o in pack.options(u, 1.0)] == [Resolution.FULL]


def test_render_cites_path_and_lines() -> None:
    u = _unit(
        path="ratelimit/bucket.py",
        qualified_name="TokenBucket.consume",
        span=Span(start_line=10, end_line=24),
        validity=0.3,
    )
    head = pack.render(u, Resolution.FULL).splitlines()[0]
    assert head.startswith("### ratelimit/bucket.py:10-24 TokenBucket.consume")
    assert "changed since written" in head
    gone = u.model_copy(update={"labels": {"tombstoned": True}})
    assert "deleted from the repo" in pack.header(gone)


def test_lexical_query_splits_identifiers_and_drops_stopwords() -> None:
    assert lexical_terms("Create a getUserById for the HTTPServer") == [
        "get",
        "user",
        "id",
        "getuserbyid",
        "http",
        "server",
        "httpserver",
    ]
    assert lexical_query("the and of") is None
    assert lexical_query("rate limiter") == "rate | limiter"
