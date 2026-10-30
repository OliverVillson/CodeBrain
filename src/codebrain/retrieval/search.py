"""Stage 1 index searches and candidate loading (sketch §7).

Three indexes: the summary view s_u and code view c_u in ``unit_vectors``
(cosine, exact scan until an HNSW index exists per model), and the lexical view
on ``units.lexical`` ranked with ``ts_rank_cd`` as the BM25 stand-in.

``ts_rank_cd`` has no IDF, so terms that occur in a large share of the searched units
are dropped from the lexical query (``term_stats`` keeps per-repo document
frequencies), and the rarest terms get a second, AND-ed lexical ranking. Identifiers
in the query (``TokenBucket.take``, ``MAX_RETRIES``) also get an exact-match ranking,
``symbol_hits``: the lexical index splits them into common words.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
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


def lexical_query(text: str, drop: Collection[str] = ()) -> str | None:
    """An OR ``tsquery`` over the query's terms, or ``None`` if nothing is left.

    OR rather than AND because a prompt is a description, not a keyword list;
    ``ts_rank_cd`` rewards units that match more of it. Terms in ``drop`` (too common to
    tell units apart) are left out unless that would leave nothing.
    """
    terms = lexical_terms(text)
    kept = [t for t in terms if t not in drop] or terms
    return " | ".join(kept) if kept else None


def rare_query(terms: Sequence[str], df: Mapping[str, int], n: int = 2) -> str | None:
    """An AND ``tsquery`` over the ``n`` rarest terms that occur at all, or ``None``."""
    present = sorted((t for t in terms if df.get(t, 0) > 0), key=lambda t: (df[t], t))
    return " & ".join(present[:n]) if len(present) >= n else None


@dataclass(frozen=True)
class TermStats:
    units: int
    df: dict[str, int]
    """Units whose lexical view contains each term."""

    def common(self, terms: Iterable[str], max_df: float) -> set[str]:
        limit = max_df * self.units
        return {t for t in terms if self.df.get(t, 0) > limit}


def stats_version(conn: psycopg.Connection, repo: str | None) -> tuple[int, str]:
    """Changes whenever units in ``repo`` are added, removed or rewritten."""
    row = conn.execute(
        "SELECT count(*), coalesce(max(updated_at)::text, '') FROM units "
        "WHERE %(repo)s::text IS NULL OR repo = %(repo)s",
        {"repo": repo},
    ).fetchone()
    return (int(row[0]), str(row[1])) if row else (0, "")


def term_stats(conn: psycopg.Connection, repo: str | None) -> TermStats:
    """Document frequency of every lexical term in ``repo`` (all units for ``None``)."""
    inner: sql.Composable = sql.SQL("SELECT lexical FROM units")
    if repo is not None:
        inner = sql.SQL("SELECT lexical FROM units WHERE repo = {}").format(sql.Literal(repo))
    q = sql.SQL("SELECT word, ndoc FROM ts_stat({})").format(sql.Literal(inner.as_string(conn)))
    df = {str(w): int(n) for w, n in conn.execute(q).fetchall()}
    units = stats_version(conn, repo)[0]
    return TermStats(units=units, df=df)


# Identifier-looking tokens: dotted paths, snake_case, camelCase, ALL_CAPS (but not
# short acronyms), names with digits, anything in backticks, and capitalized words that
# do not start a sentence.
_IDENT_TOKEN = re.compile(r"`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")
_SENTENCE_END = re.compile(r"[.!?:]\s*$")


def identifiers(text: str) -> list[str]:
    """Names in ``text`` that look like code, in order, deduplicated."""
    out: dict[str, None] = {}
    for m in _IDENT_TOKEN.finditer(text):
        quoted, tok = m.group(1), m.group(2)
        if quoted:
            for t in _IDENT_TOKEN.finditer(quoted):
                if t.group(2) and len(t.group(2)) >= 2:
                    out.setdefault(t.group(2).strip("."))
            continue
        tok = tok.strip(".")
        if len(tok) < 3:
            continue
        before = text[: m.start()]
        starts_sentence = not before.strip() or bool(_SENTENCE_END.search(before))
        if (
            "_" in tok.strip("_")
            or "." in tok
            or re.search(r"[a-z][A-Z]", tok)
            or re.search(r"[A-Za-z]\d", tok)
            or (tok.isupper() and len(tok) >= 5)  # not API, TTL, SQL
            or (tok[0].isupper() and len(tok) >= 4 and not starts_sentence)
        ):
            out.setdefault(tok)
    return list(out)


def identifier_words(names: Iterable[str]) -> list[str]:
    """The split parts of ``names`` (``TokenBucket.take`` -> token, bucket, take)."""
    words: dict[str, None] = {}
    for name in names:
        for part in re.split(r"[._]+", name):
            for p in _CAMEL.findall(part):
                words.setdefault(p.lower())
    return list(words)


def symbol_hits(
    conn: psycopg.Connection, names: Sequence[str], filters: Filters, k: int
) -> list[tuple[str, float]]:
    """Units that define or use the identifiers in ``names``, best first, with a score.

    A unit whose qualified name is the identifier (or ends with ``.identifier``) defines
    it (weight 3), one with ``identifier = ...`` at the start of a line assigns it (2),
    and one whose text has it as a whole word uses it (1). Each match is divided by
    ln(2 + number of units matching that identifier), so a rare name counts for more
    than one used everywhere; ties go to the smaller unit.
    """
    if not names:
        return []
    lowered = [n.lower() for n in names]
    leaf = [re.escape(n.rsplit(".", 1)[-1]) for n in names]
    where, params = filters.where()
    q = sql.SQL(
        r"""
        WITH q(lname, word) AS (
            SELECT * FROM unnest(%(lnames)s::text[], %(words)s::text[])
        ),
        m AS (
            SELECT u.id, u.tokens, q.lname,
                CASE
                    WHEN lower(u.qualified_name) = q.lname
                        OR lower(u.qualified_name) LIKE ('%%.' || q.lname) THEN 3.0
                    WHEN u.text ~ ('(^|\n)[ \t]*' || q.word || '[ \t]*(:[^\n=]*)?=($|[^=])')
                        THEN 2.0
                    ELSE 1.0
                END AS w
            FROM units u JOIN q ON (
                lower(u.qualified_name) = q.lname
                OR lower(u.qualified_name) LIKE ('%%.' || q.lname)
                OR u.text ~ ('\m' || q.word || '\M')
            )
            WHERE {where}
        ),
        s AS (
            SELECT id, tokens, w / ln(2 + count(*) OVER (PARTITION BY lname)) AS sc FROM m
        )
        SELECT id, sum(sc) AS score FROM s
        GROUP BY id, tokens
        ORDER BY score DESC, tokens, id
        LIMIT %(k)s
        """
    ).format(where=where)
    params.update(lnames=lowered, words=leaf, k=k)
    return [(r[0], float(r[1])) for r in conn.execute(q, params).fetchall()]


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
    """Top-k unit ids by ``ts_rank_cd`` on the lexical view for the ``tsquery`` ``tsq``."""
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
    semantic: float | None = None
    """cos(q, s_u), or ``None`` without a summary vector for the query's model."""
    code: float | None = None
    """cos(q, c_u), or ``None`` without a code vector for the query's model."""


def load_candidates(
    conn: psycopg.Connection,
    ids: Sequence[str],
    *,
    summary: tuple[str, Vector] | None,
    code: tuple[str, Vector] | None,
    tsq: str | None,
    filters: Filters,
) -> dict[str, CandidateRow]:
    """Load units with their cosine to the query on each view and their raw lexical rank.

    ``summary`` and ``code`` are ``(model, query vector)``. The cosines are computed in
    the database, so no candidate vector is sent back (a 4096-dim vector is ~40 kB of
    text per unit and view). Units failing ``filters`` are dropped, which matters for
    graph-expanded ids.
    """
    if not ids:
        return {}
    where, params = filters.where()
    lex = sql.SQL("coalesce(ts_rank_cd(u.lexical, {}), 0)").format(sql.SQL(_TSQUERY))
    q = sql.SQL(
        """
        SELECT {cols}, 1 - (sv.embedding <=> %(sq)s) AS s_cos,
            1 - (cv.embedding <=> %(cq)s) AS c_cos, {lex} AS lex
        FROM units u
        LEFT JOIN unit_vectors sv ON sv.unit_id = u.id AND sv.view = 'summary'
            AND sv.model = %(s_model)s AND sv.dim = %(s_dim)s
        LEFT JOIN unit_vectors cv ON cv.unit_id = u.id AND cv.view = 'code'
            AND cv.model = %(c_model)s AND cv.dim = %(c_dim)s
        WHERE u.id = ANY(%(ids)s) AND {where}
        """
    ).format(cols=sql.SQL(unit_columns("u")), lex=lex, where=where)
    # A view without a query vector joins nothing (model NULL) and gets no cosine.
    for key, view in (("s", summary), ("c", code)):
        model, vec = view if view is not None else (None, None)
        params[f"{key}_model"] = model
        params[f"{key}_dim"] = int(vec.shape[0]) if vec is not None else None
        params[f"{key}q"] = vec if vec is not None else np.zeros(1, dtype=np.float32)
    params.update(ids=list(ids), tsq=tsq)
    out: dict[str, CandidateRow] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        for row in cur.execute(q, params).fetchall():
            s_cos, c_cos, lex_rank = row.pop("s_cos"), row.pop("c_cos"), row.pop("lex")
            unit = row_to_unit(row)
            out[unit.id] = CandidateRow(
                unit=unit,
                lexical=float(lex_rank),
                semantic=_finite(s_cos),
                code=_finite(c_cos),
            )
    return out


def _finite(x: Any) -> float | None:
    """A cosine from pgvector, or ``None`` when there was no vector (or it was zero)."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else 0.0
