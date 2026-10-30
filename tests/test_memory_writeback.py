from datetime import UTC, datetime, timedelta

import pytest

from codebrain.config import WriteBackParams
from codebrain.memory import (
    DecisionRecord,
    EpisodeRecorder,
    Outcome,
    TaskRecord,
    parse_diff,
)
from codebrain.memory.credit import is_referenced, reference_names
from codebrain.models import (
    VALIDITY_CURRENT,
    VALIDITY_DEPRECATED,
    Edge,
    EdgeType,
    Namespace,
    Span,
    Unit,
    UnitKind,
    code_unit_id,
)
from tests.memory_fakes import BagOfWordsEmbedder, FakeStore

REPO = "demo"
PATH = "ratelimit/bucket.py"

DIFF = """\
diff --git a/ratelimit/bucket.py b/ratelimit/bucket.py
--- a/ratelimit/bucket.py
+++ b/ratelimit/bucket.py
@@ -10,3 +10,4 @@ class TokenBucket:
     def consume(self, n):
-        return self.tokens > n
+        self.refill()
+        return self.tokens >= n

diff --git a/api/webhooks.py b/api/webhooks.py
--- a/api/webhooks.py
+++ b/api/webhooks.py
@@ -1,2 +1,3 @@
 import json
+from ratelimit.retry import retry_with_backoff

"""


def code(kind: UnitKind, name: str, start: int, end: int, path: str = PATH) -> Unit:
    return Unit(
        id=code_unit_id(REPO, path, name),
        kind=kind,
        repo=REPO,
        path=path,
        qualified_name=name,
        span=Span(start_line=start, end_line=end),
        text=f"source of {name}",
    )


@pytest.fixture
def store() -> FakeStore:
    s = FakeStore()
    for u in (
        code(UnitKind.MODULE, "ratelimit.bucket", 1, 40),
        code(UnitKind.CLASS, "TokenBucket", 3, 40),
        code(UnitKind.METHOD, "TokenBucket.consume", 10, 14),
        code(UnitKind.METHOD, "TokenBucket.refill", 16, 24),
        code(UnitKind.FUNCTION, "retry_with_backoff", 1, 20, path="ratelimit/retry.py"),
        code(UnitKind.FUNCTION, "redis_incr_limit", 1, 12, path="ratelimit/redis.py"),
    ):
        s.upsert_unit(u)
    return s


@pytest.fixture
def recorder(store: FakeStore) -> EpisodeRecorder:
    return EpisodeRecorder(store, BagOfWordsEmbedder())


def task(**kw: object) -> TaskRecord:
    base: dict[str, object] = {
        "prompt": "Throttle webhook deliveries with a token bucket",
        "repo": REPO,
        "session_id": "s1",
        "plan": "Reuse TokenBucket, refill before consume.",
        "diff": DIFF,
        "tests_passed": True,
    }
    return TaskRecord.model_validate({**base, **kw})


def uid(name: str, path: str = PATH) -> str:
    return code_unit_id(REPO, path, name)


def test_records_episode_linked_to_innermost_changed_units(
    store: FakeStore, recorder: EpisodeRecorder
) -> None:
    result = recorder.record(task())

    ep = store.units[result.episode_id]
    assert ep.kind is UnitKind.EPISODE and ep.namespace is Namespace.PROJECT
    assert not result.merged and ep.version == 1
    assert "Throttle webhook deliveries" in ep.text and "Plan:" in ep.text
    assert ep.summary and "token bucket" in ep.summary.lower()
    # The diff touches consume (lines 10-13), not refill; the class and module contain it.
    assert result.linked_unit_ids == [uid("TokenBucket.consume")]
    assert (
        Edge(src_id=uid("TokenBucket.consume"), dst_id=ep.id, type=EdgeType.PRODUCED_BY)
        in store.edges
    )
    assert ep.lexical_terms and "consume" in ep.lexical_terms.split()
    # Vector and raw evidence were written.
    assert (ep.id, "summary", "fake-bow-64") in store.vectors
    [evidence] = store.episodes
    assert evidence.unit_id == ep.id and evidence.diff == DIFF and evidence.tests_passed


def test_passing_tests_raise_episode_prior(store: FakeStore, recorder: EpisodeRecorder) -> None:
    passed = store.units[recorder.record(task()).episode_id]
    failed = store.units[
        recorder.record(
            task(
                prompt="Shard the Postgres cluster by tenant id",
                plan=None,
                diff=None,
                tests_passed=False,
                outcome=Outcome.FAILED,
            )
        ).episode_id
    ]
    assert passed.quality == pytest.approx(2 / 3)
    assert failed.quality == pytest.approx(1 / 3)
    assert failed.labels["outcome"] == "failed"


def test_near_duplicate_merges_and_keeps_both_runs(
    store: FakeStore, recorder: EpisodeRecorder
) -> None:
    first = recorder.record(task())
    second = recorder.record(task(tests_passed=False))
    assert second.merged and second.episode_id == first.episode_id and second.version == 2

    ep = store.units[first.episode_id]
    assert ep.version == 2 and ep.labels["runs"] == 2
    assert "--- run 2 ---" in ep.text
    assert [e.tests_passed for e in store.episodes] == [True, False]
    assert len([u for u in store.units.values() if u.kind is UnitKind.EPISODE]) == 1
    # One pass and one fail: (1 + 1) / (2 + 2).
    assert ep.quality == pytest.approx(0.5)


def test_different_task_is_a_new_episode(store: FakeStore, recorder: EpisodeRecorder) -> None:
    a = recorder.record(task())
    b = recorder.record(task(prompt="Add OAuth login to the admin dashboard", plan=None, diff=None))
    assert not b.merged and a.episode_id != b.episode_id


def test_dup_threshold_is_configurable(store: FakeStore) -> None:
    strict = EpisodeRecorder(
        store, BagOfWordsEmbedder(), params=WriteBackParams(dup_threshold=1.01)
    )
    a = strict.record(task())
    b = strict.record(task())
    assert a.episode_id != b.episode_id


def test_decisions_are_units_and_supersede_old_ones(
    store: FakeStore, recorder: EpisodeRecorder
) -> None:
    old = recorder.record(
        task(
            prompt="Limit API calls with Redis INCR",
            diff=None,
            decisions=[DecisionRecord(text="Use Redis for all rate limits")],
        )
    )
    [old_decision] = old.decision_ids

    new = recorder.record(
        task(
            decisions=[
                DecisionRecord(
                    text="Use Redis only for cross-process limits; in-process uses TokenBucket",
                    rationale="Redis round trips dominated webhook latency",
                    applies_to=[uid("TokenBucket")],
                    supersedes=[old_decision],
                )
            ]
        )
    )
    [decision_id] = new.decision_ids
    d = store.units[decision_id]
    assert d.kind is UnitKind.DECISION and d.namespace is Namespace.PROJECT
    assert "Why: Redis round trips" in d.text
    assert d.labels["applies_to"] == [uid("TokenBucket")]
    assert Edge(src_id=d.id, dst_id=new.episode_id, type=EdgeType.PRODUCED_BY) in store.edges
    assert new.deprecated_ids == [old_decision]
    assert store.units[old_decision].validity == VALIDITY_DEPRECATED
    assert d.validity == VALIDITY_CURRENT


def test_repeated_decision_merges(store: FakeStore, recorder: EpisodeRecorder) -> None:
    rec = DecisionRecord(text="Keep rate limit state in process memory")
    a = recorder.record(task(decisions=[rec]))
    b = recorder.record(task(prompt="Add OAuth login to the admin dashboard", decisions=[rec]))
    assert a.decision_ids == b.decision_ids
    d = store.units[a.decision_ids[0]]
    assert d.version == 2 and d.labels["episodes"] == [a.episode_id, b.episode_id]


def test_usage_credit(store: FakeStore, recorder: EpisodeRecorder) -> None:
    past = store.units[
        recorder.record(task(prompt="Earlier: webhook retries", session_id="old")).episode_id
    ]
    store.retrievals.clear()

    consume, retry, redis = (
        uid("TokenBucket.consume"),
        uid("retry_with_backoff", "ratelimit/retry.py"),
        uid("redis_incr_limit", "ratelimit/redis.py"),
    )
    for u in (consume, retry, redis, past.id):
        store.log(u, "s1")
    store.log(consume, "s1", query="second recall")
    store.log(redis, "other-session")

    result = recorder.record(task(used_unit_ids=[past.id]))

    # consume was edited, retry_with_backoff imported, the past episode cited; redis unused.
    assert set(result.credit.used) == {consume, retry, past.id}
    assert result.credit.unused == [redis]
    assert result.credit.retrievals == 5
    assert store.units[consume].used_count == 2 and store.units[consume].retrieved_count == 2
    assert store.units[retry].used_count == 1
    assert store.units[redis].used_count == 0 and store.units[redis].retrieved_count == 2
    assert store.units[redis].quality < store.units[retry].quality
    # Retrievals got their training labels; the other session's is untouched.
    assert [r.used for r in store.retrievals] == [True, True, False, True, True, None]

    # Recording again does not credit the same retrievals twice.
    again = recorder.record(task())
    assert again.credit.retrievals == 0
    assert store.units[consume].used_count == 2


def test_credit_only_counts_retrievals_since_task_start(
    store: FakeStore, recorder: EpisodeRecorder
) -> None:
    consume = uid("TokenBucket.consume")
    store.log(consume, "s1")
    start = datetime.now(UTC) + timedelta(seconds=1)
    result = recorder.record(task(started_at=start))
    assert result.credit.retrievals == 0


def test_reference_names() -> None:
    method = code(UnitKind.METHOD, "TokenBucket.consume", 1, 2)
    module = code(UnitKind.MODULE, "ratelimit.bucket", 1, 2)
    assert reference_names(method) == ["consume"]
    assert "ratelimit.bucket" in reference_names(module)
    assert is_referenced(method, "bucket.consume(1)", 3)
    assert not is_referenced(method, "consumer = 1", 3)
    assert is_referenced(module, "from app.ratelimit.bucket import TokenBucket", 3)
    short = code(UnitKind.FUNCTION, "f", 1, 2)
    assert not is_referenced(short, "f(x)", 3)


def test_parse_diff() -> None:
    files = parse_diff(DIFF)
    assert [f.path for f in files] == [PATH, "api/webhooks.py"]
    bucket, hooks = files
    assert bucket.ranges() == [(11, 12)]
    assert bucket.removed == ["        return self.tokens > n"]
    assert hooks.added == ["from ratelimit.retry import retry_with_backoff"]
    assert hooks.ranges() == [(2, 2)]


def test_parse_diff_deleted_file() -> None:
    diff = "--- a/old.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-def gone():\n-    pass\n"
    [f] = parse_diff(diff)
    assert f.path == "old.py" and f.deleted and f.ranges() == [(1, 2)]


def test_agent_written_summary_skips_distiller(store: FakeStore, recorder: EpisodeRecorder) -> None:
    result = recorder.record(
        task(
            summary="Webhook throttling via TokenBucket\nRefill before consume; 429 on empty.",
            capabilities=["rate limiting"],
            touched_unit_ids=["u_missing"],
            used_unit_ids=["u_also_missing"],
        )
    )
    ep = store.units[result.episode_id]
    assert ep.summary and ep.summary.startswith("Webhook throttling via TokenBucket")
    assert "429 on empty" in ep.summary
    assert ep.capabilities == ["rate limiting"]
    assert result.unknown_unit_ids == ["u_missing", "u_also_missing"]


def test_session_only_episode(store: FakeStore, recorder: EpisodeRecorder) -> None:
    result = recorder.record(task(session_only=True, diff=None))
    assert store.units[result.episode_id].namespace is Namespace.SESSION
    with pytest.raises(ValueError):
        task(session_only=True, session_id=None)


def test_standalone_decision(store: FakeStore, recorder: EpisodeRecorder) -> None:
    decision_id, deprecated = recorder.record_decision(
        DecisionRecord(text="All public endpoints are rate limited"), repo=REPO
    )
    d = store.units[decision_id]
    assert d.kind is UnitKind.DECISION and d.repo == REPO and deprecated == []
    assert not any(e.src_id == decision_id for e in store.edges)
