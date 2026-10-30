"""Basic reads and writes for the core tables.

This is the shared write path that ingest, write-back and paging all use. Search
and ranking queries belong in ``codebrain.retrieval``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from codebrain.models import Edge, Episode, Retrieval, Span, Unit, UnitVector

UNIT_COLUMNS = (
    "id, kind, namespace, repo, session_id, path, qualified_name, language, start_line, "
    "end_line, parent_id, text, summary, signature, lexical_terms, ast_hash, tokens, layer, "
    "layer_confidence, capabilities, labels, validity, retrieved_count, used_count, version, "
    "created_at, updated_at"
)


def unit_columns(alias: str) -> str:
    """``UNIT_COLUMNS`` qualified with a table alias, for queries that join ``units``."""
    return ", ".join(f"{alias}.{c}" for c in UNIT_COLUMNS.split(", "))


def _unit_params(u: Unit) -> dict[str, Any]:
    data = u.model_dump(exclude={"span"})
    data["start_line"] = u.span.start_line if u.span else None
    data["end_line"] = u.span.end_line if u.span else None
    data["labels"] = Jsonb(u.labels)
    return data


def upsert_unit(conn: psycopg.Connection, unit: Unit) -> None:
    """Insert a unit, or overwrite the stored one with the same id.

    Usage counters are kept from the stored row: they belong to the unit's history,
    not to the latest parse.
    """
    conn.execute(
        """
        INSERT INTO units (id, kind, namespace, repo, session_id, path, qualified_name,
            language, start_line, end_line, parent_id, text, summary, signature, lexical_terms,
            ast_hash, tokens, layer, layer_confidence, capabilities, labels, validity,
            retrieved_count, used_count, version, created_at, updated_at)
        VALUES (%(id)s, %(kind)s, %(namespace)s, %(repo)s, %(session_id)s, %(path)s,
            %(qualified_name)s, %(language)s, %(start_line)s, %(end_line)s, %(parent_id)s,
            %(text)s, %(summary)s, %(signature)s, %(lexical_terms)s, %(ast_hash)s, %(tokens)s,
            %(layer)s, %(layer_confidence)s, %(capabilities)s, %(labels)s, %(validity)s,
            %(retrieved_count)s, %(used_count)s, %(version)s, %(created_at)s, %(updated_at)s)
        ON CONFLICT (id) DO UPDATE SET
            kind = EXCLUDED.kind, namespace = EXCLUDED.namespace, repo = EXCLUDED.repo,
            session_id = EXCLUDED.session_id, path = EXCLUDED.path,
            qualified_name = EXCLUDED.qualified_name, language = EXCLUDED.language,
            start_line = EXCLUDED.start_line, end_line = EXCLUDED.end_line,
            parent_id = EXCLUDED.parent_id, text = EXCLUDED.text, summary = EXCLUDED.summary,
            signature = EXCLUDED.signature, lexical_terms = EXCLUDED.lexical_terms,
            ast_hash = EXCLUDED.ast_hash, tokens = EXCLUDED.tokens, layer = EXCLUDED.layer,
            layer_confidence = EXCLUDED.layer_confidence, capabilities = EXCLUDED.capabilities,
            labels = EXCLUDED.labels, validity = EXCLUDED.validity,
            version = EXCLUDED.version, updated_at = now()
        """,
        _unit_params(unit),
    )


def row_to_unit(row: dict[str, Any]) -> Unit:
    """Build a ``Unit`` from a ``dict_row`` holding ``UNIT_COLUMNS``. Consumes the row."""
    start, end = row.pop("start_line"), row.pop("end_line")
    row["span"] = Span(start_line=start, end_line=end) if start is not None else None
    return Unit.model_validate(row)


def get_unit(conn: psycopg.Connection, unit_id: str) -> Unit | None:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(f"SELECT {UNIT_COLUMNS} FROM units WHERE id = %s", (unit_id,)).fetchone()
    return row_to_unit(row) if row else None


def delete_unit(conn: psycopg.Connection, unit_id: str) -> None:
    conn.execute("DELETE FROM units WHERE id = %s", (unit_id,))


def upsert_vector(conn: psycopg.Connection, vec: UnitVector) -> None:
    conn.execute(
        """
        INSERT INTO unit_vectors (unit_id, view, model, dim, embedding, content_hash, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (unit_id, view, model) DO UPDATE SET
            dim = EXCLUDED.dim, embedding = EXCLUDED.embedding,
            content_hash = EXCLUDED.content_hash, created_at = EXCLUDED.created_at
        """,
        (
            vec.unit_id,
            vec.view,
            vec.model,
            vec.dim,
            np.asarray(vec.embedding, dtype=np.float32),
            vec.content_hash,
            vec.created_at,
        ),
    )


def get_vectors(conn: psycopg.Connection, unit_id: str) -> list[UnitVector]:
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT unit_id, view, model, embedding, content_hash, created_at "
            "FROM unit_vectors WHERE unit_id = %s ORDER BY view, model",
            (unit_id,),
        ).fetchall()
    for r in rows:
        emb = r["embedding"]
        r["embedding"] = emb.to_list() if hasattr(emb, "to_list") else [float(x) for x in emb]
    return [UnitVector.model_validate(r) for r in rows]


def add_edges(conn: psycopg.Connection, edges: Iterable[Edge]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO edges (src_id, dst_id, type, weight) VALUES (%s, %s, %s, %s)
            ON CONFLICT (src_id, dst_id, type) DO UPDATE SET weight = EXCLUDED.weight
            """,
            [(e.src_id, e.dst_id, e.type, e.weight) for e in edges],
        )


def edges_from(conn: psycopg.Connection, unit_id: str) -> list[Edge]:
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT src_id, dst_id, type, weight FROM edges WHERE src_id = %s ORDER BY dst_id",
            (unit_id,),
        ).fetchall()
    return [Edge.model_validate(r) for r in rows]


def insert_episode(conn: psycopg.Connection, episode: Episode) -> None:
    """Store episode evidence. The episode ``Unit`` must already exist."""
    data = episode.model_dump()
    data["test_results"] = Jsonb(episode.test_results)
    conn.execute(
        """
        INSERT INTO episodes (unit_id, session_id, prompt, plan, diff, test_results,
            tests_passed, pitfalls, started_at, ended_at)
        VALUES (%(unit_id)s, %(session_id)s, %(prompt)s, %(plan)s, %(diff)s, %(test_results)s,
            %(tests_passed)s, %(pitfalls)s, %(started_at)s, %(ended_at)s)
        """,
        data,
    )


def log_retrievals(conn: psycopg.Connection, retrievals: Iterable[Retrieval]) -> None:
    """Record recalled units and bump each unit's ``retrieved_count``."""
    items = list(retrievals)
    if not items:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO retrievals (query, unit_id, session_id, rank, score, resolution,
                features, used, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    r.query,
                    r.unit_id,
                    r.session_id,
                    r.rank,
                    r.score,
                    r.resolution,
                    Jsonb(r.features),
                    r.used,
                    r.created_at,
                )
                for r in items
            ],
        )
        cur.execute(
            "UPDATE units SET retrieved_count = retrieved_count + 1 WHERE id = ANY(%s)",
            ([r.unit_id for r in items],),
        )


def credit_used(conn: psycopg.Connection, unit_ids: Iterable[str]) -> None:
    """Bump ``used_count`` for units that made it into the final diff (sketch §8)."""
    conn.execute(
        "UPDATE units SET used_count = used_count + 1 WHERE id = ANY(%s)", (list(unit_ids),)
    )


# Write-back (sketch §8) ---------------------------------------------------------------


def get_units(conn: psycopg.Connection, unit_ids: Iterable[str]) -> dict[str, Unit]:
    ids = list(dict.fromkeys(unit_ids))
    if not ids:
        return {}
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            f"SELECT {UNIT_COLUMNS} FROM units WHERE id = ANY(%s)", (ids,)
        ).fetchall()
    return {u.id: u for u in map(row_to_unit, rows)}


def episodes_for(conn: psycopg.Connection, unit_id: str) -> list[Episode]:
    """Every run recorded against an episode unit, oldest first."""
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT unit_id, session_id, prompt, plan, diff, test_results, tests_passed, "
            "pitfalls, started_at, ended_at FROM episodes WHERE unit_id = %s ORDER BY ended_at",
            (unit_id,),
        ).fetchall()
    return [Episode.model_validate(r) for r in rows]


def open_retrievals(
    conn: psycopg.Connection, session_id: str, since: datetime | None = None
) -> list[Retrieval]:
    """Retrievals in a session that have not been judged used or unused yet."""
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT id, query, unit_id, session_id, rank, score, resolution, features, used, "
            "created_at FROM retrievals WHERE session_id = %s AND used IS NULL "
            "AND (%s::timestamptz IS NULL OR created_at >= %s) ORDER BY id",
            (session_id, since, since),
        ).fetchall()
    return [Retrieval.model_validate(r) for r in rows]


def mark_retrievals(conn: psycopg.Connection, used: dict[int, bool]) -> None:
    """Set the ``used`` label on logged retrievals, the training label for weights."""
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE retrievals SET used = %s WHERE id = %s", [(v, k) for k, v in used.items()]
        )


def add_usage(conn: psycopg.Connection, unit_id: str, *, retrieved: int, used: int) -> None:
    """Add to a unit's counters. ``used_count`` never passes ``retrieved_count``."""
    conn.execute(
        """
        UPDATE units SET retrieved_count = retrieved_count + %(r)s,
            used_count = LEAST(used_count + %(u)s, retrieved_count + %(r)s)
        WHERE id = %(id)s
        """,
        {"id": unit_id, "r": retrieved, "u": used},
    )


def set_validity(conn: psycopg.Connection, unit_ids: Iterable[str], validity: float) -> None:
    conn.execute(
        "UPDATE units SET validity = %s, updated_at = now() WHERE id = ANY(%s)",
        (validity, list(unit_ids)),
    )
