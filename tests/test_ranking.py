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
    r = S.retrieval
    total = r.w_semantic + r.w_code + r.w_lexical + r.w_graph + r.w_symbol + r.w_path
    relevance = (0.35 * 0.76 + 0.15 * 0.72 + 0.15 * 0.55 + 0.10 * 0.60) / total
    prior = (r.quality_floor + (1 - r.quality_floor) * 0.8) * 1.0  # fresh: F = 1
    assert b.relevance == pytest.approx(relevance)
    assert b.prior == pytest.approx(prior)
    assert b.score == pytest.approx(prior * relevance)
    assert sum(b.contributions.values()) == pytest.approx(b.relevance)


def test_priors_scale_relevance_instead_of_adding_to_it() -> None:
    kw = dict(semantic=0.0, code=0.0, lexical=0.0, graph=0.0, now=NOW, params=S.retrieval)
    # Irrelevant scores 0 however well used: the old additive priors gave it ~0.175.
    assert score(_unit(retrieved_count=9, used_count=9), paging=S.paging, **kw).score == 0  # type: ignore[arg-type]
    hit = dict(kw, semantic=0.8, lexical=1.0)
    used = score(_unit(retrieved_count=9, used_count=9), paging=S.paging, **hit)  # type: ignore[arg-type]
    unused = score(_unit(retrieved_count=9, used_count=0), paging=S.paging, **hit)  # type: ignore[arg-type]
    assert 0 < unused.score < used.score <= used.relevance


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
    big = _unit(id="big", signature="def big()", summary="Does the big thing.", text="w " * 400)
    small = _unit(id="small", signature="def small()", summary="Small.", text="w " * 50)
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


def test_pack_prices_every_resolution_on_one_scale() -> None:
    # A short unit with a long summary: its full text is the cheaper view, whatever
    # ``unit.tokens`` says (it came from another counter).
    u = _unit(
        text="def f(): return 1", summary="Returns one. " * 30, signature="def f()", tokens=900
    )
    by_res = {o.resolution: o.tokens for o in pack.options(u, 1.0)}
    for r, t in by_res.items():
        assert t == pack.estimate_tokens(pack.render(u, r))
    assert by_res[Resolution.FULL] < by_res[Resolution.SUMMARY]
    chosen = pack.pack({"u": pack.options(u, 1.0)}, by_res[Resolution.FULL])
    assert chosen["u"].resolution is Resolution.FULL


def test_pagerank_with_zero_weight_edges_has_no_nan() -> None:
    edges = [
        Edge(src_id="a", dst_id="b", type=EdgeType.CALLS, weight=0.0),
        Edge(src_id="a", dst_id="c", type=EdgeType.CALLS),
    ]
    r = personalized_pagerank({"b": 1.0}, edges, beta=0.5)
    assert all(v == v for v in r.values())  # no NaN
    assert r["b"] == pytest.approx(1.0) and r["c"] == 0.0


def test_pack_depth_first_gives_leading_units_full_code() -> None:
    big = _unit(id="big", text="x = 1\n" * 260, signature="def big()", summary="Big.", tokens=390)
    small = _unit(id="small", text="y = 2\n" * 30, signature="def small()", summary="S.", tokens=45)
    cands = {"big": pack.options(big, 0.9), "small": pack.options(small, 0.5)}
    full = {u: next(o.tokens for o in cands[u] if o.resolution is Resolution.FULL) for u in cands}
    budget = full["big"] + full["small"] - 1
    # By value per token the knapsack summarises "big" to afford "small" in full.
    assert pack.pack(cands, budget)["big"].resolution is not Resolution.FULL

    depth = pack.pack_depth_first(cands, budget, full_first=1)
    assert depth["big"].resolution is Resolution.FULL
    assert depth["small"].resolution is not Resolution.FULL  # what is left goes to the knapsack
    assert sum(o.tokens for o in depth.values()) <= budget

    # A leading unit whose full text cannot fit is packed smaller instead of skipped.
    tight = pack.pack_depth_first(cands, full["big"] - 1, full_first=1)
    assert tight["big"].resolution is not Resolution.FULL
    assert pack.pack_depth_first(cands, budget, full_first=0) == pack.pack(cands, budget)


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


def test_identifiers_pick_out_code_names() -> None:
    from codebrain.retrieval.search import identifier_words, identifiers

    q = (
        "About the rq package: If an RQ Retry has fewer intervals, which interval does "
        "Retry.get_interval use? What are MAX_RETRIES, getUserById and `lease` for the API?"
    )
    assert identifiers(q) == ["Retry", "Retry.get_interval", "MAX_RETRIES", "getUserById", "lease"]
    # Short acronyms and the word that starts a sentence are prose.
    assert identifiers("What does the API do? Huey uses Redis.") == ["Redis"]
    assert identifier_words(["TokenBucket.take", "MAX_RETRIES"]) == [
        "token",
        "bucket",
        "take",
        "max",
        "retries",
    ]


def test_lexical_query_drops_common_terms_and_ands_the_rare_ones() -> None:
    from codebrain.retrieval.search import TermStats, rare_query

    stats = TermStats(units=100, df={"job": 90, "queue": 40, "retry": 10, "jitter": 2})
    terms = lexical_terms("job queue retry jitter")
    common = stats.common(terms, 0.2)
    assert common == {"job", "queue"}
    assert lexical_query("job queue retry jitter", drop=common) == "retry | jitter"
    assert lexical_query("job queue", drop=common) == "job | queue"  # never empty
    assert rare_query(terms, stats.df) == "jitter & retry"
    assert rare_query(["jitter", "unseen"], stats.df) is None


def _code_unit(uid: str, start: int, end: int, **kw: object) -> Unit:
    text = "\n".join(f"line_{i} = {i}" for i in range(start, end + 1))
    base: dict[str, object] = dict(
        id=uid,
        kind=UnitKind.FUNCTION,
        path="m.py",
        span=Span(start_line=start, end_line=end),
        text=text,
        summary=f"Summary of {uid}.",
        signature=f"def {uid}()",
        tokens=pack.estimate_tokens(text),
        updated_at=NOW,
    )
    base.update(kw)
    return Unit.model_validate(base)


def test_contains_needs_the_outer_full_text() -> None:
    cls = _code_unit("cls", 1, 40, kind=UnitKind.CLASS)
    method = _code_unit("m", 5, 12, kind=UnitKind.METHOD)
    assert pack.contains(cls, method) and not pack.contains(method, cls)
    for label in ("skeleton", "truncated", "split", "body"):
        assert not pack.contains(cls.model_copy(update={"labels": {label: True}}), method)
    assert not pack.contains(cls.model_copy(update={"path": "other.py"}), method)


def test_plan_skips_units_already_inside_a_full_pick() -> None:
    cls = _code_unit("cls", 1, 40, kind=UnitKind.CLASS)
    method = _code_unit("m", 5, 12, kind=UnitKind.METHOD)
    other = _code_unit("other", 50, 60)
    got = pack.plan([(cls, 1.0), (method, 0.9), (other, 0.5)], 10_000, full_first=3)
    assert set(got) == {"cls", "other"}  # the method is shown inside its class
    # The class after its method replaces it and gets its tokens back.
    got = pack.plan([(method, 1.0), (cls, 0.9)], 10_000, full_first=2)
    assert set(got) == {"cls"} and got["cls"].resolution is Resolution.FULL


def test_plan_caps_full_code_and_lists_signatures_of_the_rest() -> None:
    picks = [(_code_unit(f"u{i}", 100 * i + 1, 100 * i + 30), 1.0 - i / 100) for i in range(30)]
    full = pack.options(picks[0][0], 1.0)[-1].tokens
    budget = 4 * full
    got = pack.plan(picks, budget, full_first=10, full_share=0.5, toc_share=0.25)
    assert sum(o.tokens for o in got.values()) <= budget
    by_res = [got[u.id].resolution for u, _ in picks if u.id in got]
    assert by_res[:2] == [Resolution.FULL, Resolution.FULL]
    assert by_res.count(Resolution.FULL) == 2  # 50% of the budget
    assert by_res[-1] is Resolution.SIGNATURE  # the table of contents comes last
    assert len(got) > 10


def test_render_grouped_puts_each_file_under_one_heading() -> None:
    a1 = _code_unit("a1", 30, 31, path="a.py", qualified_name="late")
    a2 = _code_unit("a2", 1, 2, path="a.py", qualified_name="early")
    b = _code_unit("b", 1, 2, path="b.py", qualified_name="bee")
    text = pack.render_grouped(
        [(a1, Resolution.FULL), (b, Resolution.SUMMARY), (a2, Resolution.SIGNATURE)]
    )
    assert text == (
        "### a.py\n- 1-2: def a2()\n#### 30-31 late\nline_30 = 30\nline_31 = 31"
        "\n\n### b.py\n#### 1-2 bee · summary\nSummary of b."
    )


def test_skeleton_elisions_collapse_and_code_has_no_kind_tag() -> None:
    u = _unit(text="class A:\n    ...\n    ...\n    x = 1\n    ...", qualified_name="A")
    assert pack.body(u, Resolution.FULL) == "class A:\n    ...\n    x = 1\n    ..."
    assert pack.header(u) == "### a.py A"
    ep = _unit(kind=UnitKind.EPISODE, path=None, id="e1", namespace=Namespace.PROJECT)
    assert pack.header(ep) == "### e1 [episode]"


def test_dampen_hubs_scales_big_containers() -> None:
    from codebrain.retrieval.graph import GraphEdge, dampen_hubs

    edges = [GraphEdge("mod", f"f{i}", EdgeType.CONTAINS) for i in range(80)]
    edges.append(GraphEdge("f0", "f1", EdgeType.CALLS))
    out = dampen_hubs(edges, hub_degree=40)
    assert out[0].effective_weight == pytest.approx(0.5 * 40 / 80)
    assert out[-1].effective_weight == 1.0
    small = dampen_hubs(edges[:10], hub_degree=40)
    assert all(e.effective_weight == 0.5 for e in small)


def test_encode_query_embeds_once_per_model() -> None:
    from codebrain.models import VectorView
    from codebrain.retrieval.encoders import encode_query

    calls: list[str] = []

    class Enc:
        def __init__(self, model: str) -> None:
            self.model = model

        def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(self.model)
            return [[1.0, 0.0]]

    shared = Enc("m")
    out = encode_query({VectorView.SUMMARY: shared, VectorView.CODE: Enc("m")}, "q")
    assert calls == ["m"] and out[VectorView.SUMMARY][1] is out[VectorView.CODE][1]
    encode_query({VectorView.SUMMARY: Enc("a"), VectorView.CODE: Enc("b")}, "q")
    assert calls == ["m", "a", "b"]
