"""``MemoryStore`` on Postgres. Writes go through ``store.repo``.

The two reads that are not plain lookups (the duplicate check and diff-to-unit
mapping) are here rather than in ``codebrain.retrieval`` because they are exact
bookkeeping queries, not ranked search.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

import numpy as np
import psycopg

from codebrain.memory.diff import FileChange
from codebrain.models import Edge, Episode, Namespace, Retrieval, Unit, UnitKind, UnitVector
from codebrain.store import repo


class PgMemoryStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def get_units(self, unit_ids: Iterable[str]) -> dict[str, Unit]:
        return repo.get_units(self.conn, unit_ids)

    def upsert_unit(self, unit: Unit) -> None:
        repo.upsert_unit(self.conn, unit)

    def upsert_vector(self, vec: UnitVector) -> None:
        repo.upsert_vector(self.conn, vec)

    def add_edges(self, edges: Iterable[Edge]) -> None:
        items = list(edges)
        if items:
            repo.add_edges(self.conn, items)

    def insert_episode(self, episode: Episode) -> None:
        repo.insert_episode(self.conn, episode)

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
        row = self.conn.execute(
            """
            SELECT u.id, 1 - (v.embedding <=> %(q)s) AS sim
            FROM unit_vectors v JOIN units u ON u.id = v.unit_id
            WHERE u.kind = %(kind)s AND v.view = 'summary' AND v.model = %(model)s
              AND v.dim = %(dim)s AND u.repo IS NOT DISTINCT FROM %(repo)s
              AND u.namespace = %(namespace)s
              AND (u.namespace <> 'session' OR u.session_id = %(session)s)
              AND u.validity > 0 AND NOT (u.labels ? 'tombstoned')
            ORDER BY v.embedding <=> %(q)s
            LIMIT 1
            """,
            {
                "q": np.asarray(embedding, dtype=np.float32),
                "kind": kind,
                "model": model,
                "dim": len(embedding),
                "repo": repo_name,
                "namespace": namespace,
                "session": session_id,
            },
        ).fetchone()
        if row is None:
            return None
        unit = repo.get_unit(self.conn, row[0])
        return (unit, float(row[1])) if unit else None

    def units_changed_by(self, repo_name: str, changes: Sequence[FileChange]) -> list[Unit]:
        # v0 approximation: diff lines are numbered in the file as the diff leaves it, while
        # stored spans come from the last ingest, which may predate the change. Below the
        # first insertion or deletion the two drift apart, so a link can land on a
        # neighbouring unit until the next ingest. Matching the ``-`` side's old line numbers
        # for files ingested before the change would fix it.
        hits: dict[str, Unit] = {}
        for change in changes:
            ranges = change.ranges()
            rows = self.conn.execute(
                """
                SELECT u.id FROM units u
                WHERE u.repo = %s AND u.path = %s AND u.start_line IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM unnest(%s::int[], %s::int[]) AS r(lo, hi)
                      WHERE u.start_line <= r.hi AND u.end_line >= r.lo)
                """,
                (repo_name, change.path, [lo for lo, _ in ranges], [hi for _, hi in ranges]),
            ).fetchall()
            hits.update(repo.get_units(self.conn, [r[0] for r in rows]))
        return innermost(hits.values())

    def open_retrievals(self, session_id: str, since: datetime | None) -> list[Retrieval]:
        return repo.open_retrievals(self.conn, session_id, since)

    def mark_retrievals(self, used: dict[int, bool]) -> None:
        repo.mark_retrievals(self.conn, used)

    def add_usage(self, unit_id: str, *, retrieved: int, used: int) -> None:
        repo.add_usage(self.conn, unit_id, retrieved=retrieved, used=used)

    def set_validity(self, unit_ids: Iterable[str], validity: float) -> None:
        repo.set_validity(self.conn, unit_ids, validity)


def innermost(units: Iterable[Unit]) -> list[Unit]:
    """Drop units that contain another hit, so a change credits the function, not its file."""
    items = [u for u in units if u.span and u.path]
    out: list[Unit] = []
    for u in items:
        assert u.span is not None
        contains_other = any(
            o.id != u.id
            and o.path == u.path
            and o.span is not None
            and u.span.start_line <= o.span.start_line
            and o.span.end_line <= u.span.end_line
            and (o.span.end_line - o.span.start_line) < (u.span.end_line - u.span.start_line)
            for o in items
        )
        if not contains_other:
            out.append(u)
    return sorted(out, key=lambda u: (u.path or "", u.span.start_line if u.span else 0))
