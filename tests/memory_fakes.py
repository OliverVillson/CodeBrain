"""In-memory stand-ins for the store and the embedding model, for memory tests."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from codebrain.memory.diff import FileChange
from codebrain.memory.pg import innermost
from codebrain.models import (
    Edge,
    Episode,
    Namespace,
    Retrieval,
    Unit,
    UnitKind,
    UnitVector,
    VectorView,
)


class BagOfWordsEmbedder:
    """Hashes words into a small vector, so texts that share words are close."""

    model = "fake-bow-64"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(v)
        return out


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


class FakeStore:
    def __init__(self) -> None:
        self.units: dict[str, Unit] = {}
        self.vectors: dict[tuple[str, str, str], UnitVector] = {}
        self.edges: set[Edge] = set()
        self.episodes: list[Episode] = []
        self.retrievals: list[Retrieval] = []

    # helpers for tests
    def log(self, unit_id: str, session_id: str, query: str = "q") -> None:
        """What recall does: log the retrieval and bump retrieved_count."""
        self.retrievals.append(
            Retrieval(
                id=len(self.retrievals) + 1,
                query=query,
                unit_id=unit_id,
                session_id=session_id,
                rank=1,
                score=0.5,
                resolution="full",
            )
        )
        u = self.units[unit_id]
        self.units[unit_id] = u.model_copy(update={"retrieved_count": u.retrieved_count + 1})

    # MemoryStore
    def get_units(self, unit_ids: Iterable[str]) -> dict[str, Unit]:
        return {i: self.units[i] for i in unit_ids if i in self.units}

    def upsert_unit(self, unit: Unit) -> None:
        old = self.units.get(unit.id)
        if old:
            unit = unit.model_copy(
                update={"retrieved_count": old.retrieved_count, "used_count": old.used_count}
            )
        self.units[unit.id] = unit

    def upsert_vector(self, vec: UnitVector) -> None:
        assert vec.unit_id in self.units
        self.vectors[(vec.unit_id, vec.view, vec.model)] = vec

    def add_edges(self, edges: Iterable[Edge]) -> None:
        for e in edges:
            assert e.src_id in self.units and e.dst_id in self.units
            self.edges.add(e)

    def insert_episode(self, episode: Episode) -> None:
        assert episode.unit_id in self.units
        self.episodes.append(episode)

    def nearest(
        self,
        kind: UnitKind,
        model: str,
        embedding: Sequence[float],
        repo_name: str | None,
        *,
        namespace: Namespace,
        session_id: str | None = None,
    ) -> tuple[Unit, float] | None:
        best: tuple[Unit, float] | None = None
        for (uid, view, m), vec in self.vectors.items():
            u = self.units[uid]
            if u.kind is not kind or view != VectorView.SUMMARY or m != model:
                continue
            if u.repo != repo_name or vec.dim != len(embedding):
                continue
            if u.namespace is not namespace or u.validity <= 0 or u.labels.get("tombstoned"):
                continue
            if u.namespace is Namespace.SESSION and u.session_id != session_id:
                continue
            sim = _cos(vec.embedding, embedding)
            if best is None or sim > best[1]:
                best = (u, sim)
        return best

    def units_changed_by(self, repo_name: str, changes: Sequence[FileChange]) -> list[Unit]:
        hits = []
        for c in changes:
            for u in self.units.values():
                if u.repo != repo_name or u.path != c.path or u.span is None:
                    continue
                if any(u.span.start_line <= hi and u.span.end_line >= lo for lo, hi in c.ranges()):
                    hits.append(u)
        return innermost(hits)

    def open_retrievals(self, session_id: str, since: datetime | None) -> list[Retrieval]:
        return [
            r
            for r in self.retrievals
            if r.session_id == session_id
            and r.used is None
            and (since is None or r.created_at >= since)
        ]

    def mark_retrievals(self, used: dict[int, bool]) -> None:
        self.retrievals = [
            r.model_copy(update={"used": used[r.id]}) if r.id in used else r
            for r in self.retrievals
        ]

    def add_usage(self, unit_id: str, *, retrieved: int, used: int) -> None:
        u = self.units[unit_id]
        r = u.retrieved_count + retrieved
        self.units[unit_id] = u.model_copy(
            update={"retrieved_count": r, "used_count": min(u.used_count + used, r)}
        )

    def set_validity(self, unit_ids: Iterable[str], validity: float) -> None:
        for i in unit_ids:
            self.units[i] = self.units[i].model_copy(
                update={"validity": validity, "updated_at": datetime.now(UTC)}
            )
