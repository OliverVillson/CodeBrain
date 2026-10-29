"""Stage 1 index searches and candidate loading (sketch §7).

Three indexes: the summary view s_u and code view c_u in ``unit_vectors``
(cosine, exact scan until an HNSW index exists per model), and the lexical view
on ``units.lexical`` ranked with ``ts_rank_cd`` as the BM25 stand-in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from codebrain.models import Layer, Namespace, Unit, UnitKind, VectorView
from codebrain.retrieval.encoders import Vector
from codebrain.store.repo import row_to_unit, unit_columns


class Filters(BaseModel):
    """Metadata filters applied to every index and to graph-expanded units.

    Session units are only visible to their own session: they match when
    ``session_id`` equals theirs and never otherwise.
    """

    repo: str | None = None
    path_prefix: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    """Match units tagged with at least one of these."""
    languages: list[str] = Field(default_factory=list)
    kinds: list[UnitKind] = Field(default_factory=list)
    layers: list[Layer] = Field(default_factory=list)
    namespaces: list[Namespace] = Field(default_factory=list)
    session_id: str | None = None
    min_validity: float = Field(default=0.0, ge=0, le=1)

    def where(self) -> tuple[sql.Composable, dict[str, Any]]:
        """SQL predicate over the ``units`` alias ``u`` and its parameters."""
        parts: list[sql.Composable] = [
            sql.SQL("(u.namespace <> 'session' OR u.session_id = %(f_session)s)"),
            sql.SQL("u.validity >= %(f_validity)s"),
        ]
        params: dict[str, Any] = {"f_session": self.session_id, "f_validity": self.min_validity}
        if self.repo is not None:
            parts.append(sql.SQL("u.repo = %(f_repo)s"))
            params["f_repo"] = self.repo
        if self.path_prefix is not None:
            parts.append(sql.SQL("starts_with(u.path, %(f_path)s)"))
            params["f_path"] = self.path_prefix
        if self.capabilities:
            parts.append(sql.SQL("u.capabilities && %(f_caps)s::text[]"))
            params["f_caps"] = self.capabilities
        for col, values in (
            ("language", self.languages),
            ("kind", [str(k) for k in self.kinds]),
            ("layer", [str(x) for x in self.layers]),
            ("namespace", [str(n) for n in self.namespaces]),
        ):
            if values:
                key = f"f_{col}"
                parts.append(
                    sql.SQL("u.{}::text = ANY(%({})s)").format(sql.Identifier(col), sql.SQL(key))
                )
                params[key] = values
        return sql.SQL(" AND ").join(parts), params


# Dropped from lexical queries: they match every summary and carry no signal.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "create",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "make",
        "me",
        "my",
        "new",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "so",
        "that",
        "the",
        "then",
        "this",
        "to",
        "use",
        "using",
        "we",
        "what",
        "when",
        "where",
        "which",
        "why",
        "will",
        "with",
        "write",
        "you",
    ]
)
_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def lexical_terms(text: str) -> list[str]:
    """Split identifiers the way ingest does (``getUserById`` -> get, user, by, id)."""
    terms: list[str] = []
    for word in _WORD.findall(text):
        pieces = [p.lower() for p in _CAMEL.findall(word)]
        if len(pieces) > 1:
            pieces.append(word.lower())
        terms.extend(pieces)
    seen: dict[str, None] = {}
    for t in terms:
        if len(t) > 1 and t not in _STOPWORDS:
            seen.setdefault(t)
    return list(seen)


def lexical_query(text: str) -> str | None:
    """An OR ``tsquery`` over the query's terms, or ``None`` if nothing is left.

    OR rather than AND because a prompt is a description, not a keyword list;
    ``ts_rank_cd`` rewards units that match more of it.
    """
    terms = lexical_terms(text)
    return " | ".join(terms) if terms else None


# ``lexical`` mixes a 'simple' part (identifiers) and an 'english' part (summary),
# so the query is built in both configurations and OR-ed.
_TSQUERY = "(to_tsquery('simple', %(tsq)s) || to_tsquery('english', %(tsq)s))"


def vector_hits(
    conn: psycopg.Connection,
    view: VectorView,
    model: str,
    query: Vector,
    filters: Filters,
    k: int,
) -> list[str]:
    """Top-k unit ids by cosine distance on one dense view."""
    where, params = filters.where()
    q = sql.SQL(
        """
        SELECT u.id FROM unit_vectors v JOIN units u ON u.id = v.unit_id
        WHERE v.view = %(view)s AND v.model = %(model)s AND v.dim = %(dim)s AND {where}
        ORDER BY v.embedding <=> %(q)s, u.id
        LIMIT %(k)s
        """
    ).format(where=where)
    params.update(view=view, model=model, dim=int(query.shape[0]), q=query, k=k)
    return [r[0] for r in conn.execute(q, params).fetchall()]


def lexical_hits(conn: psycopg.Connection, tsq: str | None, filters: Filters, k: int) -> list[str]:
    """Top-k unit ids by ``ts_rank_cd`` on the lexical view."""
    if tsq is None:
        return []
    where, params = filters.where()
    q = sql.SQL(
        """
        SELECT u.id FROM units u
        WHERE u.lexical @@ {tsquery} AND {where}
        ORDER BY ts_rank_cd(u.lexical, {tsquery}) DESC, u.id
        LIMIT %(k)s
        """
    ).format(tsquery=sql.SQL(_TSQUERY), where=where)
    params.update(tsq=tsq, k=k)
    return [r[0] for r in conn.execute(q, params).fetchall()]


@dataclass
class CandidateRow:
    """A unit with the raw inputs stage 2 needs."""

    unit: Unit
    lexical: float
    summary_vec: Vector | None
    code_vec: Vector | None


def load_candidates(
    conn: psycopg.Connection,
    ids: Sequence[str],
    *,
    summary_model: str | None,
    code_model: str | None,
    tsq: str | None,
    filters: Filters,
) -> dict[str, CandidateRow]:
    """Load units with their vectors for the query's models and their raw lexical rank.

    Units failing ``filters`` are dropped, which matters for graph-expanded ids.
    """
    if not ids:
        return {}
    where, params = filters.where()
    lex = sql.SQL("coalesce(ts_rank_cd(u.lexical, {}), 0)").format(sql.SQL(_TSQUERY))
    q = sql.SQL(
        """
        SELECT {cols}, sv.embedding AS s_emb, cv.embedding AS c_emb, {lex} AS lex
        FROM units u
        LEFT JOIN unit_vectors sv
            ON sv.unit_id = u.id AND sv.view = 'summary' AND sv.model = %(s_model)s
        LEFT JOIN unit_vectors cv
            ON cv.unit_id = u.id AND cv.view = 'code' AND cv.model = %(c_model)s
        WHERE u.id = ANY(%(ids)s) AND {where}
        """
    ).format(cols=sql.SQL(unit_columns("u")), lex=lex, where=where)
    params.update(ids=list(ids), s_model=summary_model, c_model=code_model, tsq=tsq)
    out: dict[str, CandidateRow] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        for row in cur.execute(q, params).fetchall():
            s_emb, c_emb, lex_rank = row.pop("s_emb"), row.pop("c_emb"), row.pop("lex")
            unit = row_to_unit(row)
            out[unit.id] = CandidateRow(
                unit=unit,
                lexical=float(lex_rank),
                summary_vec=_as_vec(s_emb),
                code_vec=_as_vec(c_emb),
            )
    return out


def _as_vec(emb: Any) -> Vector | None:
    if emb is None:
        return None
    # pgvector >= 0.4 returns ``Vector``; older versions return an ndarray.
    arr = emb.to_numpy() if hasattr(emb, "to_numpy") else emb
    return np.asarray(arr, dtype=np.float32)
