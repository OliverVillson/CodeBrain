import textwrap
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codebrain.config import ChunkingParams, Settings
from codebrain.ingest import Ingestor, MemoryStore, PostgresStore
from codebrain.ingest.embed import HashingEmbedder
from codebrain.ingest.pipeline import discover
from codebrain.ingest.store import IngestStore
from codebrain.ingest.summarize import HeuristicSummarizer, SummaryRequest
from codebrain.models import (
    VALIDITY_CHANGED,
    Edge,
    EdgeType,
    Layer,
    Namespace,
    Unit,
    UnitKind,
    VectorView,
    code_unit_id,
)
from codebrain.store import repo as db

REPO = "demo"
CLOCK = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)

BUCKET = '''\
"""Rate limiting."""
from .clock import now


class TokenBucket:
    """Refills at a fixed rate; refuses when empty."""

    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = now()

    def consume(self, n=1):
        self.refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def refill(self):
        t = now()
        self.tokens = min(self.capacity, self.tokens + (t - self.last) * self.rate)
        self.last = t
'''

CLOCK_PY = "def now():\n    import time\n    return time.monotonic()\n"


class CountingSummarizer(HeuristicSummarizer):
    name = "counting-v1"

    def __init__(self, fail: Sequence[str] = ()) -> None:
        self.seen: list[str] = []
        self.fail = set(fail)

    def summarize(self, requests: Sequence[SummaryRequest]) -> list[str | None]:
        self.seen.extend(r.qualified_name for r in requests)
        return [None if r.qualified_name in self.fail else self.one(r) for r in requests]


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest) -> IngestStore:
    if request.param == "memory":
        return MemoryStore()
    return PostgresStore(request.getfixturevalue("conn"))


def write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))


def make(store: IngestStore, summarizer: CountingSummarizer | None = None) -> Ingestor:
    settings = Settings(chunking=ChunkingParams(tau_min=5, tau_max=1200))
    emb = HashingEmbedder(64)
    return Ingestor(
        store,
        REPO,
        settings=settings,
        summarizer=summarizer or CountingSummarizer(),
        embedders={VectorView.SUMMARY: emb, VectorView.CODE: emb},
        clock=lambda: CLOCK,
    )


def uid(path: str, qn: str) -> str:
    return code_unit_id(REPO, path, qn)


def unit(store: IngestStore, path: str, qn: str) -> Unit:
    found = [u for u in store.units_at(REPO, path) if u.qualified_name == qn]
    assert found, f"no unit {path}:{qn}"
    return found[0]


def edges_from(store: IngestStore, src: str) -> set[tuple[str, EdgeType]]:
    if isinstance(store, MemoryStore):
        return {(e.dst_id, e.type) for e in store.edges.values() if e.src_id == src}
    assert isinstance(store, PostgresStore)
    return {(e.dst_id, e.type) for e in db.edges_from(store.conn, src)}


def test_full_ingest_writes_units_vectors_and_edges(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/bucket.py": BUCKET, "rl/clock.py": CLOCK_PY, "notes.txt": "skip me"})
    r = make(store).ingest(tmp_path)
    assert r.errors == []
    assert (r.files_seen, r.files_parsed) == (2, 2)
    assert r.units_written == 7  # 2 modules, class, 3 methods, now()

    consume = unit(store, "rl/bucket.py", "TokenBucket.consume")
    assert consume.kind is UnitKind.METHOD and consume.namespace is Namespace.CODE
    assert consume.parent_id == uid("rl/bucket.py", "TokenBucket")
    assert consume.span is not None and (consume.span.start_line, consume.span.end_line) == (14, 19)
    assert consume.summary and consume.summary.startswith("Method TokenBucket.consume")
    assert consume.lexical_terms and "token" in consume.lexical_terms.split()
    assert consume.labels == {"summarizer": "counting-v1"}
    assert consume.tokens > 0 and consume.ast_hash

    module = unit(store, "rl/bucket.py", "rl/bucket.py")
    assert module.labels["file_hash"] and module.labels["ingest_version"] == 1
    # Parent summaries are written from the children's summaries.
    assert "consume" in (unit(store, "rl/bucket.py", "TokenBucket").summary or "")

    cid = uid("rl/bucket.py", "TokenBucket.consume")
    assert store.vector_hash(cid, VectorView.SUMMARY, "hashing-v1-64")
    assert store.vector_hash(cid, VectorView.CODE, "hashing-v1-64")
    assert edges_from(store, cid) == {(uid("rl/bucket.py", "TokenBucket.refill"), EdgeType.CALLS)}
    assert (uid("rl/clock.py", "now"), EdgeType.CALLS) in edges_from(
        store, uid("rl/bucket.py", "TokenBucket.refill")
    )
    assert (uid("rl/clock.py", "rl/clock.py"), EdgeType.IMPORTS) in edges_from(
        store, uid("rl/bucket.py", "rl/bucket.py")
    )


def test_rerun_skips_unchanged_files(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/bucket.py": BUCKET, "rl/clock.py": CLOCK_PY})
    s = CountingSummarizer()
    make(store, s).ingest(tmp_path)
    s.seen.clear()
    r = make(store, s).ingest(tmp_path)
    assert (r.files_unchanged, r.files_parsed, r.units_written) == (2, 0, 0)
    assert s.seen == []


def test_edit_resummarizes_only_changed_units(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/bucket.py": BUCKET, "rl/clock.py": CLOCK_PY})
    s = CountingSummarizer()
    make(store, s).ingest(tmp_path)
    before = unit(store, "rl/bucket.py", "TokenBucket.refill")
    s.seen.clear()

    edited = BUCKET.replace("self.tokens -= n", "self.tokens -= 2 * n").replace(
        '"""Rate limiting."""', '"""Rate limiting."""\n# a new comment shifts every line'
    )
    write(tmp_path, {"rl/bucket.py": edited})
    r = make(store, s).ingest(tmp_path, ["rl/bucket.py"])

    # The edited method and its ancestors are re-summarized; siblings are not.
    assert sorted(s.seen) == sorted(["TokenBucket.consume", "TokenBucket", "rl/bucket.py"])
    assert r.summaries_reused == 2
    assert r.units_unchanged == 2
    consume = unit(store, "rl/bucket.py", "TokenBucket.consume")
    assert consume.version == 2
    refill = unit(store, "rl/bucket.py", "TokenBucket.refill")
    assert refill.version == 1 and refill.ast_hash == before.ast_hash
    assert refill.span is not None and before.span is not None
    assert refill.span.start_line == before.span.start_line + 1  # lines follow the file
    assert r.vectors_embedded > 0 and r.vectors_reused > 0


def test_removed_function_is_tombstoned_and_revived(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/bucket.py": BUCKET, "rl/clock.py": CLOCK_PY})
    make(store).ingest(tmp_path)
    refill_id = uid("rl/bucket.py", "TokenBucket.refill")

    without = BUCKET.split("    def refill")[0].replace("        self.refill()\n", "")
    write(tmp_path, {"rl/bucket.py": without})
    r = make(store).ingest(tmp_path)
    assert r.units_removed == 1
    gone = unit(store, "rl/bucket.py", "TokenBucket.refill")
    assert gone.labels["tombstoned"] is True
    assert gone.labels["tombstoned_at"] == CLOCK.isoformat()
    assert gone.validity == VALIDITY_CHANGED
    assert edges_from(store, refill_id) == set()
    assert edges_from(store, uid("rl/bucket.py", "TokenBucket.consume")) == set()
    assert refill_id not in {s.symbol.id for s in store.symbols(REPO)}

    write(tmp_path, {"rl/bucket.py": BUCKET})
    r = make(store).ingest(tmp_path)
    back = unit(store, "rl/bucket.py", "TokenBucket.refill")
    assert "tombstoned" not in back.labels and back.validity == 1.0
    assert r.units_removed == 0


def test_deleted_file_is_tombstoned(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/bucket.py": BUCKET, "rl/clock.py": CLOCK_PY})
    make(store).ingest(tmp_path)
    (tmp_path / "rl/clock.py").unlink()
    r = make(store).ingest(tmp_path, ["rl/clock.py"])
    assert (r.files_removed, r.units_removed) == (1, 2)
    assert all(u.labels.get("tombstoned") for u in store.units_at(REPO, "rl/clock.py"))
    # The caller's edge to the deleted function went with the tombstone.
    assert (uid("rl/clock.py", "now"), EdgeType.CALLS) not in edges_from(
        store, uid("rl/bucket.py", "TokenBucket.refill")
    )
    # A second run has nothing left to remove.
    assert make(store).ingest(tmp_path).units_removed == 0


def test_moved_code_reuses_vectors(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"a/clock.py": CLOCK_PY})
    make(store).ingest(tmp_path)
    (tmp_path / "a/clock.py").unlink()
    write(tmp_path, {"b/clock.py": CLOCK_PY})
    r = make(store).ingest(tmp_path)
    assert r.units_removed == 2
    # The code view of ``now`` is unchanged by the move, so its vector is copied.
    assert r.vectors_reused >= 1
    assert store.vector_hash(uid("b/clock.py", "now"), VectorView.CODE, "hashing-v1-64")


def test_linked_episodes_and_decisions_go_stale(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/bucket.py": BUCKET, "rl/clock.py": CLOCK_PY})
    make(store).ingest(tmp_path)
    episode = Unit(kind=UnitKind.EPISODE, namespace=Namespace.PROJECT, text="Built a bucket")
    decision = Unit(kind=UnitKind.DECISION, namespace=Namespace.PROJECT, text="Use a bucket")
    noted = Unit(
        kind=UnitKind.DECISION,
        namespace=Namespace.PROJECT,
        text="Clock is monotonic",
        labels={"applies_to": [uid("rl/clock.py", "now")]},
    )
    for u in (episode, decision, noted):
        store.upsert_unit(u)
    consume = uid("rl/bucket.py", "TokenBucket.consume")
    store.replace_edges(
        [decision.id],
        [Edge(src_id=decision.id, dst_id=episode.id, type=EdgeType.PRODUCED_BY)],
    )
    store.replace_edges(
        [consume],
        [
            Edge(src_id=consume, dst_id=episode.id, type=EdgeType.PRODUCED_BY),
            Edge(
                src_id=consume,
                dst_id=uid("rl/bucket.py", "TokenBucket.refill"),
                type=EdgeType.CALLS,
            ),
        ],
    )

    write(tmp_path, {"rl/bucket.py": BUCKET.replace("self.tokens -= n", "self.tokens -= 3 * n")})
    r = make(store).ingest(tmp_path)
    assert r.linked_marked_stale == 2
    (tmp_path / "rl/clock.py").unlink()
    r = make(store).ingest(tmp_path)
    assert r.linked_marked_stale == 1

    def validity(u: Unit) -> float:
        if isinstance(store, MemoryStore):
            return store.units[u.id].validity
        assert isinstance(store, PostgresStore)
        got = db.get_unit(store.conn, u.id)
        assert got is not None
        return got.validity

    assert [validity(u) for u in (episode, decision, noted)] == [VALIDITY_CHANGED] * 3
    # The produced_by edge is not structural, so ingest leaves it in place.
    assert (episode.id, EdgeType.PRODUCED_BY) in edges_from(store, consume)


def test_classification_and_foreign_labels_survive(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/clock.py": CLOCK_PY})
    make(store).ingest(tmp_path)
    now = unit(store, "rl/clock.py", "now")
    store.upsert_unit(
        now.model_copy(
            update={
                "layer": Layer.INFRA,
                "layer_confidence": 0.9,
                "capabilities": ["timekeeping"],
                "labels": {**now.labels, "visibility": "public"},
            }
        )
    )
    write(tmp_path, {"rl/clock.py": CLOCK_PY.replace("monotonic", "time")})
    make(store).ingest(tmp_path)
    after = unit(store, "rl/clock.py", "now")
    assert after.layer is Layer.INFRA and after.capabilities == ["timekeeping"]
    assert after.labels["visibility"] == "public"
    assert after.version == 2


def test_failed_summary_falls_back_and_retries(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/clock.py": CLOCK_PY})
    make(store, CountingSummarizer(fail=["now"])).ingest(tmp_path)
    now = unit(store, "rl/clock.py", "now")
    assert now.summary and now.labels["summarizer"] == "heuristic-v1"
    s = CountingSummarizer()
    make(store, s).ingest(tmp_path, force=True)
    assert "now" in s.seen
    assert unit(store, "rl/clock.py", "now").labels["summarizer"] == "counting-v1"


def test_force_redoes_summaries_and_vectors(tmp_path: Path, store: IngestStore) -> None:
    write(tmp_path, {"rl/clock.py": CLOCK_PY})
    make(store).ingest(tmp_path)
    r = make(store).ingest(tmp_path, force=True)
    assert r.files_parsed == 1 and r.summaries_reused == 0
    assert r.vectors_embedded == 4 and r.vectors_reused == 0
    assert r.linked_marked_stale == 0 and r.units_unchanged == 2


def test_prompt_files_become_prompt_units(tmp_path: Path, store: IngestStore) -> None:
    write(
        tmp_path, {"prompts/review.md": "# Review\nYou review pull requests.\n", "README.md": "x"}
    )
    make(store).ingest(tmp_path)
    [p] = store.units_at(REPO, "prompts/review.md")
    assert p.kind is UnitKind.PROMPT and p.language is None
    assert store.units_at(REPO, "README.md") == []


def test_discover_respects_excludes_and_binary(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "src/a.py": "x = 1\n",
            "node_modules/lib/index.js": "x",
            ".hidden/b.py": "x = 1\n",
            "docs/CLAUDE.md": "rules",
            "src/blob.py": "",
        },
    )
    (tmp_path / "src/blob.py").write_bytes(b"\x00\x01")
    assert discover(tmp_path, ChunkingParams()) == ["docs/CLAUDE.md", "src/a.py"]
