"""What ingest needs from the memory store, with Postgres and in-memory versions.

Writes of units, vectors and edges go through ``codebrain.store.repo``; the
queries here are the ingest-specific reads (units at a path, vectors by content
hash) and the bulk updates for tombstoning.

Tombstones: a code unit whose code is gone keeps its row, so episodes and
decisions that point at it still resolve. It gets ``validity = 0.3`` (sketch §7:
"code changed or gone, lessons may still apply") and ``labels.tombstoned = true``
with ``labels.tombstoned_at``, and loses its structural edges. Retrieval can filter
on ``labels ? 'tombstoned'``. If the same id shows up again, re-ingest overwrites
the row and it is live again.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import psycopg
from psycopg.rows import dict_row

from codebrain.ingest.graph import STRUCTURAL_EDGES, Symbol
from codebrain.models import (
    VALIDITY_CHANGED,
    Edge,
    EdgeType,
    Namespace,
    Unit,
    UnitKind,
    UnitVector,
    VectorView,
)
from codebrain.store import repo as db

TOMBSTONE_LABEL = "tombstoned"


def is_tombstoned(unit: Unit) -> bool:
    return bool(unit.labels.get(TOMBSTONE_LABEL))


@dataclass(frozen=True)
class StoredSymbol:
    symbol: Symbol
    aliases: tuple[str, ...] = ()


class IngestStore(Protocol):
    def symbols(self, repo: str) -> list[StoredSymbol]:
        """Live (not tombstoned) code units of ``repo``."""
        ...

    def units_at(self, repo: str, path: str) -> list[Unit]:
        """Every code unit stored for ``path``, tombstoned ones included."""
        ...

    def upsert_unit(self, unit: Unit) -> None: ...

    def vector_hash(self, unit_id: str, view: VectorView, model: str) -> str | None: ...

    def find_vector(self, view: VectorView, model: str, content_hash: str) -> list[float] | None:
        """Any stored embedding of the same input, to reuse instead of re-embedding."""
        ...

    def upsert_vector(self, vec: UnitVector) -> None: ...

    def replace_edges(self, src_ids: Sequence[str], edges: Iterable[Edge]) -> None:
        """Drop the structural edges leaving ``src_ids``, then add ``edges``."""
        ...

    def tombstone(self, unit_ids: Sequence[str], at: datetime) -> None: ...

    def mark_linked_stale(self, unit_ids: Sequence[str]) -> int:
        """Lower to 0.3 the validity of episodes and decisions tied to these code units.

        Episodes: ``produced_by`` edges from the code unit (code -> episode).
        Decisions: edges from a decision to one of those episodes, or a decision whose
        ``labels.applies_to`` lists the code unit.
        """
        ...


_STRUCTURAL = [t.value for t in STRUCTURAL_EDGES]


class PostgresStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def symbols(self, repo: str) -> list[StoredSymbol]:
        rows = self.conn.execute(
            """
            SELECT id, path, qualified_name, kind, labels -> 'merged'
            FROM units
            WHERE repo = %s AND namespace = 'code' AND path IS NOT NULL
              AND qualified_name IS NOT NULL AND NOT (labels ? 'tombstoned')
            """,
            (repo,),
        ).fetchall()
        return [
            StoredSymbol(Symbol(r[0], r[1], r[2], UnitKind(r[3])), tuple(r[4] or ())) for r in rows
        ]

    def units_at(self, repo: str, path: str) -> list[Unit]:
        with self.conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                f"SELECT {db.UNIT_COLUMNS} FROM units "
                "WHERE repo = %s AND path = %s AND namespace = 'code' ORDER BY start_line, id",
                (repo, path),
            ).fetchall()
        return [db.row_to_unit(r) for r in rows]

    def upsert_unit(self, unit: Unit) -> None:
        db.upsert_unit(self.conn, unit)

    def vector_hash(self, unit_id: str, view: VectorView, model: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM unit_vectors WHERE unit_id = %s AND view = %s AND model = %s",
            (unit_id, view, model),
        ).fetchone()
        return row[0] if row else None

    def find_vector(self, view: VectorView, model: str, content_hash: str) -> list[float] | None:
        row = self.conn.execute(
            "SELECT embedding FROM unit_vectors WHERE model = %s AND view = %s "
            "AND content_hash = %s LIMIT 1",
            (model, view, content_hash),
        ).fetchone()
        if row is None:
            return None
        emb = row[0]
        return emb.to_list() if hasattr(emb, "to_list") else [float(x) for x in emb]

    def upsert_vector(self, vec: UnitVector) -> None:
        db.upsert_vector(self.conn, vec)

    def replace_edges(self, src_ids: Sequence[str], edges: Iterable[Edge]) -> None:
        self.conn.execute(
            "DELETE FROM edges WHERE src_id = ANY(%s) AND type = ANY(%s::edge_type[])",
            (list(src_ids), _STRUCTURAL),
        )
        db.add_edges(self.conn, edges)

    def tombstone(self, unit_ids: Sequence[str], at: datetime) -> None:
        if not unit_ids:
            return
        ids = list(unit_ids)
        self.conn.execute(
            """
            UPDATE units SET
                validity = LEAST(validity, %s),
                labels = labels
                    || jsonb_build_object('tombstoned', true, 'tombstoned_at', %s::text),
                updated_at = now()
            WHERE id = ANY(%s)
            """,
            (VALIDITY_CHANGED, at.isoformat(), ids),
        )
        self.conn.execute(
            "DELETE FROM edges WHERE (src_id = ANY(%s) OR dst_id = ANY(%s)) "
            "AND type = ANY(%s::edge_type[])",
            (ids, ids, _STRUCTURAL),
        )

    def mark_linked_stale(self, unit_ids: Sequence[str]) -> int:
        if not unit_ids:
            return 0
        ids = list(unit_ids)
        cur = self.conn.execute(
            """
            WITH eps AS (
                SELECT e.dst_id AS id FROM edges e JOIN units u ON u.id = e.dst_id
                WHERE e.type = 'produced_by' AND e.src_id = ANY(%(ids)s) AND u.kind = 'episode'
            ), decs AS (
                SELECT e.src_id AS id FROM edges e JOIN units u ON u.id = e.src_id
                WHERE e.dst_id IN (SELECT id FROM eps) AND u.kind = 'decision'
                UNION
                SELECT id FROM units
                WHERE kind = 'decision' AND labels -> 'applies_to' ?| %(ids)s
            )
            UPDATE units SET validity = %(v)s, updated_at = now()
            WHERE id IN (SELECT id FROM eps UNION SELECT id FROM decs) AND validity > %(v)s
            """,
            {"ids": ids, "v": VALIDITY_CHANGED},
        )
        return cur.rowcount


@dataclass
class MemoryStore:
    """Dict-backed store with the same behaviour, for tests and dry runs."""

    units: dict[str, Unit] = field(default_factory=dict)
    vectors: dict[tuple[str, VectorView, str], UnitVector] = field(default_factory=dict)
    edges: dict[tuple[str, str, EdgeType], Edge] = field(default_factory=dict)

    def symbols(self, repo: str) -> list[StoredSymbol]:
        return [
            StoredSymbol(
                Symbol(u.id, u.path, u.qualified_name, u.kind), tuple(u.labels.get("merged", ()))
            )
            for u in self.units.values()
            if u.repo == repo
            and u.namespace is Namespace.CODE
            and u.path
            and u.qualified_name
            and not is_tombstoned(u)
        ]

    def units_at(self, repo: str, path: str) -> list[Unit]:
        return [
            u
            for u in self.units.values()
            if u.repo == repo and u.path == path and u.namespace is Namespace.CODE
        ]

    def upsert_unit(self, unit: Unit) -> None:
        old = self.units.get(unit.id)
        if old is not None:  # usage counters belong to the stored row, as in repo.upsert_unit
            unit = unit.model_copy(
                update={"retrieved_count": old.retrieved_count, "used_count": old.used_count}
            )
        self.units[unit.id] = unit

    def vector_hash(self, unit_id: str, view: VectorView, model: str) -> str | None:
        v = self.vectors.get((unit_id, view, model))
        return v.content_hash if v else None

    def find_vector(self, view: VectorView, model: str, content_hash: str) -> list[float] | None:
        for (_, v_view, v_model), v in self.vectors.items():
            if v_view == view and v_model == model and v.content_hash == content_hash:
                return list(v.embedding)
        return None

    def upsert_vector(self, vec: UnitVector) -> None:
        self.vectors[(vec.unit_id, vec.view, vec.model)] = vec

    def replace_edges(self, src_ids: Sequence[str], edges: Iterable[Edge]) -> None:
        srcs = set(src_ids)
        for key in [k for k in self.edges if k[0] in srcs and k[2] in STRUCTURAL_EDGES]:
            del self.edges[key]
        for e in edges:
            self.edges[(e.src_id, e.dst_id, e.type)] = e

    def tombstone(self, unit_ids: Sequence[str], at: datetime) -> None:
        ids = set(unit_ids)
        for uid in ids:
            u = self.units.get(uid)
            if u is not None:
                labels = {**u.labels, TOMBSTONE_LABEL: True, "tombstoned_at": at.isoformat()}
                self.units[uid] = u.model_copy(
                    update={"validity": min(u.validity, VALIDITY_CHANGED), "labels": labels}
                )
        for key in [
            k for k in self.edges if (k[0] in ids or k[1] in ids) and k[2] in STRUCTURAL_EDGES
        ]:
            del self.edges[key]

    def mark_linked_stale(self, unit_ids: Sequence[str]) -> int:
        ids = set(unit_ids)

        def kind(uid: str) -> UnitKind | None:
            u = self.units.get(uid)
            return u.kind if u else None

        eps = {
            dst
            for (src, dst, t) in self.edges
            if t is EdgeType.PRODUCED_BY and src in ids and kind(dst) is UnitKind.EPISODE
        }
        decs = {
            src for (src, dst, _) in self.edges if dst in eps and kind(src) is UnitKind.DECISION
        }
        decs |= {
            u.id
            for u in self.units.values()
            if u.kind is UnitKind.DECISION and ids & set(u.labels.get("applies_to") or ())
        }
        n = 0
        for uid in eps | decs:
            u = self.units.get(uid)
            if u is not None and u.validity > VALIDITY_CHANGED:
                self.units[uid] = u.model_copy(update={"validity": VALIDITY_CHANGED})
                n += 1
        return n
