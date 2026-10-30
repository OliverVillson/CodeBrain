import psycopg
import pytest

from codebrain.models import (
    Edge,
    EdgeType,
    Episode,
    Namespace,
    Resolution,
    Retrieval,
    Span,
    Unit,
    UnitKind,
    UnitVector,
    VectorView,
    code_unit_id,
)
from codebrain.store import connect, migrate, repo
from codebrain.store.migrate import discover


def _fn(name: str = "TokenBucket.consume") -> Unit:
    return Unit(
        id=code_unit_id("demo", "ratelimit/bucket.py", name),
        kind=UnitKind.METHOD,
        repo="demo",
        path="ratelimit/bucket.py",
        qualified_name=name,
        language="python",
        span=Span(start_line=10, end_line=24),
        text="def consume(self, n): ...",
        summary="Take n tokens from the bucket, refusing when it is empty.",
        lexical_terms="token bucket consume",
        tokens=80,
        capabilities=["rate limiting"],
        labels={"visibility": "public"},
    )


def test_migrations_are_numbered() -> None:
    assert [m.version for m in discover()][0] == 1


def test_migrate_is_idempotent(database_url: str) -> None:
    assert migrate(database_url) == []


def test_unit_roundtrip(conn: psycopg.Connection) -> None:
    u = _fn()
    repo.upsert_unit(conn, u)
    got = repo.get_unit(conn, u.id)
    assert got is not None
    assert got.span == u.span
    assert got.kind is UnitKind.METHOD
    assert got.labels == {"visibility": "public"}
    assert got.capabilities == ["rate limiting"]


def test_upsert_keeps_usage_counters(conn: psycopg.Connection) -> None:
    u = _fn()
    repo.upsert_unit(conn, u)
    repo.credit_used(conn, [u.id])
    repo.upsert_unit(conn, u.model_copy(update={"summary": "changed"}))
    got = repo.get_unit(conn, u.id)
    assert got is not None and got.used_count == 1 and got.summary == "changed"


def test_lexical_index_matches(conn: psycopg.Connection) -> None:
    repo.upsert_unit(conn, _fn())
    row = conn.execute(
        "SELECT count(*) FROM units WHERE lexical @@ plainto_tsquery('simple', 'bucket')"
    ).fetchone()
    assert row is not None and row[0] == 1


def test_vectors_and_similarity(conn: psycopg.Connection) -> None:
    u = _fn()
    repo.upsert_unit(conn, u)
    repo.upsert_vector(
        conn, UnitVector(unit_id=u.id, view=VectorView.SUMMARY, model="m", embedding=[1, 0, 0])
    )
    [v] = repo.get_vectors(conn, u.id)
    assert v.dim == 3 and v.embedding == [1.0, 0.0, 0.0]
    row = conn.execute(
        "SELECT 1 - (embedding <=> '[1,0,0]') FROM unit_vectors WHERE unit_id = %s", (u.id,)
    ).fetchone()
    assert row is not None and row[0] == pytest.approx(1.0)


def test_edges_and_cascade(conn: psycopg.Connection) -> None:
    a, b = _fn("A.a"), _fn("B.b")
    repo.upsert_unit(conn, a)
    repo.upsert_unit(conn, b)
    repo.add_edges(conn, [Edge(src_id=a.id, dst_id=b.id, type=EdgeType.CALLS)])
    assert [e.dst_id for e in repo.edges_from(conn, a.id)] == [b.id]
    repo.delete_unit(conn, b.id)
    assert repo.edges_from(conn, a.id) == []


def test_episode_and_retrieval_log(conn: psycopg.Connection) -> None:
    code = _fn()
    ep = Unit(kind=UnitKind.EPISODE, namespace=Namespace.PROJECT, text="Built webhook throttling")
    repo.upsert_unit(conn, code)
    repo.upsert_unit(conn, ep)
    repo.insert_episode(conn, Episode(unit_id=ep.id, prompt="Throttle webhooks", tests_passed=True))
    repo.add_edges(conn, [Edge(src_id=code.id, dst_id=ep.id, type=EdgeType.PRODUCED_BY)])
    repo.log_retrievals(
        conn,
        [
            Retrieval(
                query="rate limiter",
                unit_id=code.id,
                rank=1,
                score=0.74,
                resolution=Resolution.FULL,
                features={"sem": 0.76},
            )
        ],
    )
    got = repo.get_unit(conn, code.id)
    assert got is not None and got.retrieved_count == 1


def test_schema_rejects_session_unit_without_session(conn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO units (id, kind, namespace, text) VALUES ('x', 'turn', 'session', 't')"
        )


@pytest.mark.db
def test_connect_does_not_leave_a_transaction_open(database_url: str) -> None:
    for autocommit in (False, True):
        with connect(database_url, autocommit=autocommit) as c:
            assert c.info.transaction_status is psycopg.pq.TransactionStatus.IDLE
            assert c.autocommit is autocommit
            # pgvector types are still registered.
            assert c.execute("SELECT '[1,2]'::vector").fetchone() is not None
