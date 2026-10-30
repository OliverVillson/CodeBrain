"""Recall and explain end to end against Postgres + pgvector.

The corpus mirrors the sketch's worked example: "Create a rate limiter for the
public API" over a token bucket, two past episodes (one whose code was deleted),
an unrelated retry helper, and middleware that only the code graph connects.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from codebrain.config import Settings
from codebrain.models import (
    Edge,
    EdgeType,
    Namespace,
    Resolution,
    Span,
    Unit,
    UnitKind,
    UnitVector,
    VectorView,
)
from codebrain.retrieval import ContextBroker, Filters
from codebrain.store import repo
from tests.fakes import HashEmbedder

NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)
QUERY = "Create a rate limiter for the public API"
TEXT = HashEmbedder("text-test")
CODE = HashEmbedder("code-test")


def _code(uid: str, path: str, name: str, summary: str, terms: str, **kw: object) -> Unit:
    data: dict[str, object] = dict(
        id=uid,
        kind=UnitKind.FUNCTION,
        repo="demo",
        path=path,
        qualified_name=name,
        language="python",
        span=Span(start_line=10, end_line=24),
        text=f"def {name.split('.')[-1]}(...):\n"
        + "".join(f"    step_{i} = run(step_{i - 1})\n" for i in range(1, 30)),
        summary=summary,
        signature=f"def {name.split('.')[-1]}(...)",
        lexical_terms=terms,
        tokens=200,
        updated_at=NOW,
        created_at=NOW,
    )
    data.update(kw)
    return Unit.model_validate(data)


def _episode(uid: str, summary: str, **kw: object) -> Unit:
    data: dict[str, object] = dict(
        id=uid,
        kind=UnitKind.EPISODE,
        namespace=Namespace.PROJECT,
        repo="demo",
        text=summary + "\n" + "Steps and pitfalls. " * 30,
        summary=summary,
        lexical_terms=summary.lower(),
        tokens=180,
        updated_at=NOW - timedelta(days=7),
        created_at=NOW - timedelta(days=7),
    )
    data.update(kw)
    return Unit.model_validate(data)


def _store(conn: psycopg.Connection, unit: Unit, *, vectors: bool = True) -> Unit:
    repo.upsert_unit(conn, unit)
    if vectors:
        for view, enc, text in (
            (VectorView.SUMMARY, TEXT, unit.summary or unit.text),
            (VectorView.CODE, CODE, f"{unit.lexical_terms or ''} {unit.text}"),
        ):
            repo.upsert_vector(
                conn,
                UnitVector(
                    unit_id=unit.id, view=view, model=enc.model, embedding=enc.embed([text])[0]
                ),
            )
    return unit


@pytest.fixture
def corpus(conn: psycopg.Connection) -> dict[str, Unit]:
    units = {
        "bucket": _store(
            conn,
            _code(
                "bucket",
                "ratelimit/bucket.py",
                "TokenBucket.consume",
                "Rate limit API requests with a token bucket; refuse when empty.",
                "token bucket consume rate limit",
                retrieved_count=3,
                used_count=3,
            ),
        ),
        "retry": _store(
            conn,
            _code(
                "retry",
                "net/retry.py",
                "retry_with_backoff",
                "Retry a failing call with exponential backoff.",
                "retry with backoff",
            ),
        ),
        "config": _store(
            conn,
            _code(
                "config",
                "app/config.py",
                "load_config",
                "Load the YAML settings file.",
                "load config yaml",
            ),
        ),
        "webhook": _store(
            conn,
            _episode("webhook", "Episode: throttled outgoing webhooks per tenant with a limiter."),
        ),
        "redis": _store(
            conn,
            _episode(
                "redis",
                "Episode: Redis INCR rate limiter for the public API.",
                validity=0.3,
                updated_at=NOW - timedelta(days=120),
            ),
        ),
        # Only reachable through the graph: no vectors, no shared terms.
        "middleware": _store(
            conn,
            _code(
                "middleware",
                "web/middleware.py",
                "quota_middleware",
                "Wrap each handler and answer 429 once over quota.",
                "quota middleware",
            ),
            vectors=False,
        ),
    }
    repo.add_edges(
        conn,
        [
            Edge(src_id="middleware", dst_id="bucket", type=EdgeType.CALLS),
            Edge(src_id="bucket", dst_id="webhook", type=EdgeType.PRODUCED_BY),
        ],
    )
    return units


def _settings() -> Settings:
    """Defaults, but keep weak hits: most tests look at where every unit ranks, and in a
    six-unit corpus the relative cutoff leaves little more than the top hit."""
    settings = Settings()
    settings.retrieval.min_relative_score = 0.0
    return settings


def _broker(conn: psycopg.Connection, settings: Settings | None = None) -> ContextBroker:
    encoders = {VectorView.SUMMARY: TEXT, VectorView.CODE: CODE}
    return ContextBroker(conn, encoders, settings=settings or _settings(), clock=lambda: NOW)


@pytest.mark.db
def test_recall_ranks_live_code_over_tombstoned_episode(
    conn: psycopg.Connection, corpus: dict[str, Unit]
) -> None:
    result = _broker(conn).recall(QUERY, 10_000, log=False)
    ids = [u.unit.id for u in result.units]
    assert ids[0] == "bucket"
    assert ids.index("redis") > ids.index("webhook")
    top = result.units[0]
    assert top.citation == "ratelimit/bucket.py:10-24"
    assert top.text.startswith("### ratelimit/bucket.py:10-24 TokenBucket.consume")
    assert result.tokens_used <= result.budget
    # The context groups units under one heading per file.
    assert result.render().startswith("### ratelimit/bucket.py\n#### 10-24 TokenBucket.consume\n")


@pytest.mark.db
def test_relative_cutoff_drops_weak_hits(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    settings = _settings()
    settings.retrieval.min_relative_score = 0.5
    result = _broker(conn, settings).recall(QUERY, 10_000, log=False)
    kept = {u.unit.id for u in result.units}
    everything = {u.unit.id for u in _broker(conn).recall(QUERY, 10_000, log=False).units}
    assert "bucket" in kept and kept < everything
    best = result.units[0].score
    assert all(u.score >= 0.5 * best for u in result.units)


@pytest.mark.db
def test_explain_breaks_down_validity(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    ex = _broker(conn).explain(QUERY, "redis", budget=10_000)
    assert ex.found and ex.candidate and ex.selected
    b = ex.breakdown
    assert b is not None
    assert set(ex.index_ranks) == {"semantic", "code", "lexical"}
    assert b.validity == pytest.approx(0.3)
    assert b.score == pytest.approx(0.3 * b.prior * b.relevance)
    assert b.semantic > 0.5  # semantically the closest unit, pulled down by validity
    assert sum(b.contributions.values()) == pytest.approx(b.relevance)


@pytest.mark.db
def test_deprecated_units_are_not_returned(
    conn: psycopg.Connection, corpus: dict[str, Unit]
) -> None:
    repo.upsert_unit(conn, corpus["redis"].model_copy(update={"validity": 0.0}))
    broker = _broker(conn)
    assert "redis" not in [u.unit.id for u in broker.recall(QUERY, 10_000, log=False).units]
    ex = broker.explain(QUERY, "redis", budget=10_000)
    assert ex.candidate and ex.breakdown is not None and ex.breakdown.score == 0
    assert ex.mmr_rank is None and ex.selected is False


@pytest.mark.db
def test_graph_expansion_pulls_in_callers(
    conn: psycopg.Connection, corpus: dict[str, Unit]
) -> None:
    broker = _broker(conn)
    ex = broker.explain(QUERY, "middleware", budget=10_000)
    assert ex.candidate and ex.graph_expanded and ex.index_ranks == {}
    assert ex.breakdown is not None and ex.breakdown.graph > 0
    assert ex.breakdown.semantic == 0 and ex.breakdown.lexical == 0
    assert "middleware" in [u.unit.id for u in broker.recall(QUERY, 10_000, log=False).units]


@pytest.mark.db
def test_small_budget_drops_resolution(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    broker = _broker(conn)
    roomy = broker.recall(QUERY, 10_000, log=False)
    assert {u.resolution for u in roomy.units} == {Resolution.FULL}
    tight = broker.recall(QUERY, 120, log=False)
    assert 0 < tight.tokens_used <= 120
    assert Resolution.FULL not in {u.resolution for u in tight.units}
    assert broker.recall(QUERY, 10_000, k=2, log=False).units[1:] != []
    assert len(broker.recall(QUERY, 10_000, k=2, log=False).units) == 2


@pytest.mark.db
def test_filters_and_session_scope(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    _store(
        conn,
        _code(
            "other", "x.py", "rate_limit", "Rate limit the public API.", "rate limit", repo="other"
        ),
    )
    turn = _store(
        conn,
        Unit(
            id="turn",
            kind=UnitKind.TURN,
            namespace=Namespace.SESSION,
            session_id="s1",
            text="User wants the public API rate limiter to allow bursts.",
            summary="User wants the public API rate limiter to allow bursts.",
            lexical_terms="public api rate limiter bursts",
            updated_at=NOW,
        ),
    )
    broker = _broker(conn)

    def ids(f: Filters) -> list[str]:
        return [u.unit.id for u in broker.recall(QUERY, 10_000, filters=f, log=False).units]

    assert "other" in ids(Filters())
    assert "other" not in ids(Filters(repo="demo"))
    assert turn.id not in ids(Filters())
    assert turn.id not in ids(Filters(session_id="s2"))
    assert turn.id in ids(Filters(session_id="s1"))
    assert ids(Filters(kinds=[UnitKind.EPISODE])) == ["webhook", "redis"]
    assert "redis" not in ids(Filters(min_validity=0.5))
    assert ids(Filters(path_prefix="ratelimit/")) == ["bucket"]
    repo.upsert_unit(conn, corpus["retry"].model_copy(update={"capabilities": ["resilience"]}))
    assert ids(Filters(capabilities=["resilience", "caching"])) == ["retry"]
    ex = broker.explain(QUERY, turn.id, filters=Filters(session_id="s1"))
    assert ex.breakdown is not None and ex.breakdown.namespace_weight == pytest.approx(1.2)


@pytest.mark.db
def test_recall_logs_retrievals(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    result = _broker(conn).recall(QUERY, 10_000, filters=Filters(session_id="s1"))
    got = repo.get_unit(conn, "bucket")
    assert got is not None and got.retrieved_count == 4
    rows = conn.execute(
        "SELECT unit_id, rank, session_id, features FROM retrievals ORDER BY rank"
    ).fetchall()
    assert [r[0] for r in rows] == [u.unit.id for u in result.units]
    assert rows[0][1] == 1 and rows[0][2] == "s1"
    assert set(rows[0][3]) >= {"semantic", "code", "lexical", "graph", "quality", "freshness"}


@pytest.mark.db
def test_explain_scores_units_outside_the_candidates(
    conn: psycopg.Connection, corpus: dict[str, Unit]
) -> None:
    settings = _settings()
    settings.retrieval.k_per_index = 1
    settings.graph.expand_keep = 0
    ex = _broker(conn, settings).explain(QUERY, "config", budget=10_000)
    assert ex.found and not ex.candidate and ex.index_ranks == {}
    # Nothing in it matches the query, so only the priors are non-zero.
    assert ex.breakdown is not None and ex.breakdown.prior > 0 and ex.breakdown.score == 0
    assert ex.selected is False
    missing = _broker(conn).explain(QUERY, "no-such-unit")
    assert not missing.found and missing.breakdown is None


@pytest.mark.db
def test_mmr_spreads_near_duplicates(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    dup = corpus["bucket"].model_copy(
        update={
            "id": "bucket_copy",
            "path": "vendor/bucket.py",
            "retrieved_count": 0,
            "used_count": 0,
        }
    )
    _store(conn, dup)
    ranked = [u.unit.id for u in _broker(conn).recall(QUERY, 10_000, log=False).units]
    assert ranked[0] == "bucket"
    assert ranked.index("bucket_copy") > 1


@pytest.mark.db
def test_lexical_only_without_encoders(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    broker = ContextBroker(conn, {}, settings=Settings(), clock=lambda: NOW)
    result = broker.recall("token bucket", 10_000, log=False)
    assert result.units[0].unit.id == "bucket"
    assert result.units[0].breakdown.semantic == 0


@pytest.mark.db
def test_graph_expansion_loads_a_bounded_prefix(
    conn: psycopg.Connection, corpus: dict[str, Unit], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bucket's module contains 200 more units: all are 1-2 hops from a seed.
    module = _store(
        conn,
        _code("mod", "ratelimit/bucket.py", "ratelimit/bucket.py", "Module.", "module"),
    )
    repo.add_edges(conn, [Edge(src_id=module.id, dst_id="bucket", type=EdgeType.CONTAINS)])
    for i in range(200):
        u = _store(
            conn,
            _code(f"n{i}", "ratelimit/bucket.py", f"n{i}", "Other code.", "zzz"),
            vectors=False,
        )
        repo.add_edges(conn, [Edge(src_id=module.id, dst_id=u.id, type=EdgeType.CONTAINS)])

    from codebrain.retrieval import broker as broker_mod

    loaded: list[int] = []
    real = broker_mod.search.load_candidates

    def counting(*args: object, **kw: object) -> object:
        loaded.append(len(args[1]))  # type: ignore[arg-type]
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(broker_mod.search, "load_candidates", counting)
    settings = _settings()
    settings.graph.expand_keep = 5
    settings.graph.expand_load_factor = 2
    result = _broker(conn, settings).recall(QUERY, 10_000, log=False)
    assert result.units
    # Seeds, then one batch of 2 * 5 neighbours was enough to fill the 5 slots.
    assert loaded[1:] == [10]


@pytest.mark.db
def test_symbol_hits_rank_definitions_over_mentions(conn: psycopg.Connection) -> None:
    from codebrain.retrieval import search

    defs = _store(conn, _code("defs", "q/defaults.py", "q/defaults.py", "Defaults.", "defaults"))
    repo.upsert_unit(conn, defs.model_copy(update={"text": "MAX_RETRIES = 7\nOTHER = 1\n"}))
    _store(conn, _code("use", "q/worker.py", "Worker.run", "Runs jobs.", "worker run"))
    use = repo.get_unit(conn, "use")
    assert use is not None
    repo.upsert_unit(conn, use.model_copy(update={"text": "def run():\n    n = MAX_RETRIES\n"}))
    _store(conn, _code("bucket", "rl/bucket.py", "TokenBucket.consume", "Bucket.", "token bucket"))

    hits = search.symbol_hits(conn, ["MAX_RETRIES", "TokenBucket.consume"], Filters(), 10)
    ids = [uid for uid, _ in hits]
    assert ids[0] == "bucket"  # defined by name, and the only unit matching it
    assert ids.index("defs") < ids.index("use")  # assigned beats mentioned
    assert search.symbol_hits(conn, ["MAX_RETRIES"], Filters(repo="nope"), 10) == []


@pytest.mark.db
def test_recall_reports_stage_timings_and_symbol_matches(
    conn: psycopg.Connection, corpus: dict[str, Unit]
) -> None:
    result = _broker(conn).recall("Where is TokenBucket.consume?", 10_000, log=False)
    assert result.units[0].unit.id == "bucket"
    assert result.units[0].breakdown.symbol == pytest.approx(1.0)
    assert {"embed", "search_lexical", "search_symbol", "graph", "load", "score", "pack"} <= set(
        result.timings
    )
    assert all(v >= 0 for v in result.timings.values())


@pytest.mark.db
def test_term_stats_follow_the_repo(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    from codebrain.retrieval import search

    stats = search.term_stats(conn, "demo")
    assert stats.units == 6
    assert stats.df["bucket"] >= 1 and "nonexistentterm" not in stats.df
    broker = _broker(conn)
    broker.recall(QUERY, 1000, log=False)
    version = search.stats_version(conn, None)
    _store(conn, _code("new", "n.py", "n", "Rate limit.", "rate limit", repo="demo"))
    assert search.stats_version(conn, None) != version
    assert search.term_stats(conn, "demo").units == 7


@pytest.mark.db
def test_reranker_boosts_its_picks(conn: psycopg.Connection, corpus: dict[str, Unit]) -> None:
    seen: list[list[str]] = []

    class PickLast:
        """Picks the weakest candidate it is shown."""

        def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[str]:
            seen.append([uid for uid, _ in candidates])
            assert all(text.startswith("### ") for _, text in candidates)
            return [candidates[-1][0]]

    settings = _settings()
    settings.retrieval.cross_encoder_top = 3
    encoders = {VectorView.SUMMARY: TEXT, VectorView.CODE: CODE}
    broker = ContextBroker(
        conn, encoders, settings=settings, clock=lambda: NOW, reranker=PickLast()
    )
    result = broker.recall(QUERY, 10_000, log=False)
    assert len(seen) == 1 and len(seen[0]) == 3
    assert result.units[0].unit.id == seen[0][-1] != "bucket"


def test_constant_values_follow_the_code_that_uses_them() -> None:
    from codebrain.retrieval import constants

    code = "def run(self):\n    t = self.timeout or CALLBACK_TIMEOUT\n    LOCAL_LIMIT = 3\n"
    assert constants.referenced([code, "x = LOCAL_LIMIT + MAX_JOBS"]) == [
        "CALLBACK_TIMEOUT",
        "MAX_JOBS",
    ]
    defaults = (
        'import os\n\n# Seconds.\nCALLBACK_TIMEOUT = 60\n"""Callback limit."""\n'
        "DELAYS = (\n    1,\n    5,\n)\nclass Q:\n    MAX_JOBS: int = 10  # per queue\n"
    )
    assert constants.statement(defaults, "CALLBACK_TIMEOUT") == (
        '# Seconds.\nCALLBACK_TIMEOUT = 60\n"""Callback limit."""'
    )
    assert constants.statement(defaults, "DELAYS") == "DELAYS = (\n    1,\n    5,\n)"
    assert constants.statement(defaults, "MAX_JOBS") == "MAX_JOBS: int = 10  # per queue"
    assert constants.statement("if CALLBACK_TIMEOUT == 3: pass", "CALLBACK_TIMEOUT") is None


@pytest.mark.db
def test_recall_adds_the_values_of_constants_the_code_uses(conn: psycopg.Connection) -> None:
    lines = "".join(f"    step_{i} = run(step_{i - 1})\n" for i in range(1, 20))
    use = _code(
        "cb",
        "q/callbacks.py",
        "Callback.__init__",
        "Callback timeout.",
        "callback timeout",
        text=f"def __init__(self, timeout=None):\n{lines}    t = timeout or CALLBACK_TIMEOUT\n",
    )
    _store(conn, use)
    for uid, path in (("near", "q/defaults.py"), ("far", "other/lib/defaults.py")):
        _store(
            conn,
            _code(
                uid,
                path,
                path,
                "Defaults.",
                "zzz",
                kind=UnitKind.MODULE,
                text=f"# Seconds.\nCALLBACK_TIMEOUT = {60 if uid == 'near' else 5}\nOTHER_X = 1\n",
            ),
            vectors=False,
        )
    filters = Filters(path_prefix="q/callbacks")
    res = _broker(conn).recall("callback timeout", 2000, filters=filters, log=False)
    assert [u.unit.id for u in res.units] == ["cb"]
    # The definition nearest to the code that uses it, and only the names it uses.
    assert [(c.name, c.path) for c in res.constants] == [("CALLBACK_TIMEOUT", "q/defaults.py")]
    text = res.render()
    assert "### q/defaults.py\n#### constants used above\n# Seconds.\nCALLBACK_TIMEOUT = 60" in text
    assert res.tokens_used == res.units[0].tokens + res.constants[0].tokens <= 2000

    # Off, or no room left: nothing added.
    settings = _settings()
    settings.retrieval.max_constants = 0
    assert _broker(conn, settings).recall("callback timeout", 2000, filters=filters).constants == []
    tight = _broker(conn).recall("callback timeout", res.units[0].tokens, filters=filters)
    assert tight.constants == []
