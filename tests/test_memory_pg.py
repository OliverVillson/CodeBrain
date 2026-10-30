"""Write-back and paging against real Postgres (skipped without CODEBRAIN_DATABASE_URL)."""

import psycopg
import pytest

from codebrain.memory import (
    DecisionRecord,
    EpisodeRecorder,
    PgMemoryStore,
    Role,
    SessionTracker,
    TaskRecord,
)
from codebrain.models import (
    VALIDITY_DEPRECATED,
    EdgeType,
    Namespace,
    Resolution,
    Retrieval,
    Span,
    Unit,
    UnitKind,
    code_unit_id,
)
from codebrain.store import repo
from tests.memory_fakes import BagOfWordsEmbedder

pytestmark = pytest.mark.db

REPO = "demo"
DIFF = """\
--- a/ratelimit/bucket.py
+++ b/ratelimit/bucket.py
@@ -10,2 +10,3 @@
     def consume(self, n):
+        self.refill()
         return self.tokens >= n
"""


def _code(kind: UnitKind, name: str, start: int, end: int) -> Unit:
    return Unit(
        id=code_unit_id(REPO, "ratelimit/bucket.py", name),
        kind=kind,
        repo=REPO,
        path="ratelimit/bucket.py",
        qualified_name=name,
        span=Span(start_line=start, end_line=end),
        text=name,
    )


@pytest.fixture
def store(conn: psycopg.Connection) -> PgMemoryStore:
    for u in (
        _code(UnitKind.MODULE, "ratelimit.bucket", 1, 40),
        _code(UnitKind.CLASS, "TokenBucket", 3, 40),
        _code(UnitKind.METHOD, "TokenBucket.consume", 10, 14),
        _code(UnitKind.METHOD, "TokenBucket.peek", 20, 24),
    ):
        repo.upsert_unit(conn, u)
    return PgMemoryStore(conn)


def _task(**kw: object) -> TaskRecord:
    base: dict[str, object] = {
        "prompt": "Throttle webhook deliveries with a token bucket",
        "repo": REPO,
        "session_id": "s1",
        "diff": DIFF,
        "tests_passed": True,
    }
    return TaskRecord.model_validate({**base, **kw})


def test_record_merge_and_credit(conn: psycopg.Connection, store: PgMemoryStore) -> None:
    consume = code_unit_id(REPO, "ratelimit/bucket.py", "TokenBucket.consume")
    peek = code_unit_id(REPO, "ratelimit/bucket.py", "TokenBucket.peek")
    repo.log_retrievals(
        conn,
        [
            Retrieval(
                query="rate limiter",
                unit_id=uid,
                session_id="s1",
                rank=i + 1,
                score=0.5,
                resolution=Resolution.FULL,
            )
            for i, uid in enumerate([consume, peek])
        ],
    )
    recorder = EpisodeRecorder(store, BagOfWordsEmbedder())
    first = recorder.record(_task(decisions=[DecisionRecord(text="Keep limiter state in process")]))

    assert first.linked_unit_ids == [consume]
    assert [e.dst_id for e in repo.edges_from(conn, consume)] == [first.episode_id]
    assert first.credit.used == [consume] and first.credit.unused == [peek]
    got = repo.get_units(conn, [consume, peek, first.episode_id])
    assert got[consume].used_count == 1 and got[consume].retrieved_count == 1
    assert got[peek].used_count == 0 and got[peek].retrieved_count == 1
    assert got[first.episode_id].quality == pytest.approx(2 / 3)
    used = conn.execute("SELECT used FROM retrievals ORDER BY id").fetchall()
    assert [r[0] for r in used] == [True, False]
    [edge] = repo.edges_from(conn, first.decision_ids[0])
    assert edge.type is EdgeType.PRODUCED_BY and edge.dst_id == first.episode_id

    second = recorder.record(_task(tests_passed=False))
    assert second.merged and second.episode_id == first.episode_id and second.version == 2
    runs = repo.episodes_for(conn, first.episode_id)
    assert [r.tests_passed for r in runs] == [True, False]

    later = recorder.record_decision(
        DecisionRecord(text="Share limiter state through Redis", supersedes=first.decision_ids),
        repo=REPO,
    )
    old = repo.get_unit(conn, first.decision_ids[0])
    assert later[1] == first.decision_ids and old and old.validity == VALIDITY_DEPRECATED


def test_paging_writes_session_turn_units(conn: psycopg.Connection, store: PgMemoryStore) -> None:
    t = SessionTracker("s2", 1000, store=store, embedder=BagOfWordsEmbedder(), repo=REPO)
    for i in range(8):
        t.add_turn(Role.USER, f"turn number {i}", tokens=100)
    assert t.state.paged
    unit = repo.get_unit(conn, t.state.paged[0].unit_id)
    assert unit and unit.kind is UnitKind.TURN and unit.namespace is Namespace.SESSION
    assert "turn number 0" in unit.text
    row = conn.execute(
        "SELECT count(*) FROM unit_vectors WHERE unit_id = %s", (unit.id,)
    ).fetchone()
    assert row and row[0] == 1
