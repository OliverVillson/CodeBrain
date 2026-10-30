"""The context broker: ``recall`` and ``explain`` (sketch §7, Figure 3).

Stages, in order:

1. Search the summary, code and lexical indexes (top ``k_per_index`` each) and
   fuse them with RRF, keeping ``fused_keep``.
2. Expand with personalized PageRank seeded by the fused RRF scores, adding up
   to ``expand_keep`` graph neighbours no index found.
3. Score every candidate with S(u | q).
4. Order by MMR, then pack into the token budget choosing each unit's
   resolution. The packing value of a unit is its MMR value, so a unit that
   mostly repeats an earlier pick is worth less than its raw score.

Intent classification, query rewriting, the hypothetical-answer draft and the
optional cross-encoder from Figure 3 are not here yet; recall embeds the query
as given.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import psycopg
from pydantic import BaseModel

from codebrain.config import Settings, get_settings
from codebrain.models import Resolution, Retrieval, Unit, VectorView
from codebrain.retrieval import fusion, graph, pack, search
from codebrain.retrieval.encoders import Encoders, Vector, cosine, encode_query
from codebrain.retrieval.mmr import Pick, mmr
from codebrain.retrieval.scoring import ScoreBreakdown, score
from codebrain.retrieval.search import CandidateRow, Filters
from codebrain.store import repo

INDEXES = ("semantic", "code", "lexical")


class RecalledUnit(BaseModel):
    unit: Unit
    rank: int
    resolution: Resolution
    tokens: int
    score: float
    breakdown: ScoreBreakdown
    citation: str
    text: str
    """The rendered block: a citation header and the unit at ``resolution``."""


class RecallResult(BaseModel):
    query: str
    budget: int
    tokens_used: int
    units: list[RecalledUnit]

    def render(self) -> str:
        """The context block to hand the agent."""
        return "\n\n".join(u.text for u in self.units)


class Explanation(BaseModel):
    """Why a unit did or did not come back for a query."""

    query: str
    unit_id: str
    found: bool
    """The unit exists and passes the filters."""
    candidate: bool
    """Reached stage 2, from an index or from graph expansion."""
    index_ranks: dict[str, int]
    rrf: float | None
    graph_expanded: bool
    breakdown: ScoreBreakdown | None
    mmr_rank: int | None
    redundancy: float | None
    selected: bool | None
    """Packed into ``budget``; ``None`` when no budget was given."""
    resolution: Resolution | None


@dataclass
class _Ranking:
    """Everything stages 1 to 3 produce, kept so ``explain`` can report on it."""

    rankings: dict[str, list[str]]
    fused: dict[str, float]
    pagerank: dict[str, float]
    expanded: set[str]
    rows: dict[str, CandidateRow]
    breakdowns: dict[str, ScoreBreakdown]
    picks: list[Pick]
    lexical_max: float
    pagerank_max: float
    now: datetime
    query_vecs: dict[VectorView, tuple[str, Vector]] = field(default_factory=dict)
    tsq: str | None = None


class ContextBroker:
    """Recall over one database connection.

    ``encoders`` maps each dense view to the model ingest used for it; views
    without one are skipped. Writes (the retrieval log) happen in ``conn``'s
    transaction, so the caller commits.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        encoders: Encoders | None = None,
        *,
        settings: Settings | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.conn = conn
        self.encoders: Encoders = encoders or {}
        self.settings = settings or get_settings()
        self.clock = clock or (lambda: datetime.now(UTC))

    def recall(
        self,
        query: str,
        budget: int,
        *,
        k: int | None = None,
        filters: Filters | None = None,
        log: bool = True,
    ) -> RecallResult:
        """Return the best context for ``query`` that fits in ``budget`` tokens.

        ``k`` caps the number of units considered for packing. With ``log``, each
        returned unit is written to ``retrievals`` and its ``retrieved_count``
        bumped (the usage-credit denominator, sketch §8).
        """
        filters = filters or Filters()
        ranking = self._rank(query, filters)
        packed, order = self._pack(ranking, budget, k)

        units: list[RecalledUnit] = []
        for uid in order:
            if uid not in packed:
                continue
            opt = packed[uid]
            unit = ranking.rows[uid].unit
            b = ranking.breakdowns[uid]
            units.append(
                RecalledUnit(
                    unit=unit,
                    rank=len(units) + 1,
                    resolution=opt.resolution,
                    tokens=opt.tokens,
                    score=b.score,
                    breakdown=b,
                    citation=pack.citation(unit),
                    text=pack.render(unit, opt.resolution),
                )
            )
        result = RecallResult(
            query=query, budget=budget, tokens_used=sum(u.tokens for u in units), units=units
        )
        if log and units:
            repo.log_retrievals(
                self.conn,
                [
                    Retrieval(
                        query=query,
                        unit_id=u.unit.id,
                        session_id=filters.session_id,
                        rank=u.rank,
                        score=u.score,
                        resolution=u.resolution,
                        features=u.breakdown.features(),
                    )
                    for u in units
                ],
            )
        return result

    def explain(
        self,
        query: str,
        unit_id: str,
        *,
        budget: int | None = None,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> Explanation:
        """Score breakdown and stage-by-stage trace of one unit for ``query``.

        A unit that never became a candidate is still scored, against the same
        normalizers, so you can see how far off it was.
        """
        filters = filters or Filters()
        ranking = self._rank(query, filters)
        breakdown = ranking.breakdowns.get(unit_id)
        candidate = unit_id in ranking.rows
        found = candidate
        if not candidate:
            row = self._load(ranking, [unit_id], filters).get(unit_id)
            if row is not None:
                found = True
                breakdown = self._score(ranking, row)

        mmr_rank = redundancy = None
        for i, p in enumerate(ranking.picks, start=1):
            if p.unit_id == unit_id:
                mmr_rank, redundancy = i, p.redundancy
                break

        selected: bool | None = None
        resolution: Resolution | None = None
        if budget is not None:
            packed, _ = self._pack(ranking, budget, k)
            selected = unit_id in packed
            resolution = packed[unit_id].resolution if selected else None

        return Explanation(
            query=query,
            unit_id=unit_id,
            found=found,
            candidate=candidate,
            index_ranks=fusion.ranks_of(ranking.rankings, unit_id),
            rrf=ranking.fused.get(unit_id),
            graph_expanded=unit_id in ranking.expanded,
            breakdown=breakdown,
            mmr_rank=mmr_rank,
            redundancy=redundancy,
            selected=selected,
            resolution=resolution,
        )

    # Stages 1 to 3.

    def _rank(self, query: str, filters: Filters) -> _Ranking:
        rp, gp = self.settings.retrieval, self.settings.graph
        qvecs = encode_query(self.encoders, query)
        tsq = search.lexical_query(query)

        rankings: dict[str, list[str]] = {}
        for name, view in (("semantic", VectorView.SUMMARY), ("code", VectorView.CODE)):
            if view in qvecs:
                model, vec = qvecs[view]
                rankings[name] = search.vector_hits(
                    self.conn, view, model, vec, filters, rp.k_per_index
                )
        rankings["lexical"] = search.lexical_hits(self.conn, tsq, filters, rp.k_per_index)

        fused = dict(list(fusion.rrf(rankings, rp.rrf_k).items())[: rp.fused_keep])
        edges = graph.neighborhood(self.conn, fused, gp.ppr_hops)
        pr = graph.personalized_pagerank(
            fused,
            edges,
            beta=gp.ppr_beta,
            iterations=gp.ppr_iterations,
            tolerance=gp.ppr_tolerance,
        )
        neighbours = sorted(
            (u for u in pr if u not in fused and pr[u] > 0), key=lambda u: (-pr[u], u)
        )

        ranking = _Ranking(
            rankings=rankings,
            fused=fused,
            pagerank=pr,
            expanded=set(),
            rows={},
            breakdowns={},
            picks=[],
            lexical_max=0.0,
            pagerank_max=max(pr.values(), default=0.0),
            now=self.clock(),
            query_vecs=qvecs,
            tsq=tsq,
        )
        rows = self._load(ranking, list(fused), filters)
        # Expansion is capped after filtering so filtered-out neighbours don't use slots,
        # but only a bounded prefix of the (often repo-sized) neighbourhood is loaded.
        kept: list[str] = []
        batch = max(1, gp.expand_load_factor * gp.expand_keep)
        start = 0
        while len(kept) < gp.expand_keep and start < len(neighbours):
            chunk = neighbours[start : start + batch]
            start += batch
            loaded = self._load(ranking, chunk, filters)
            for u in chunk:
                if u in loaded and len(kept) < gp.expand_keep:
                    kept.append(u)
                    rows[u] = loaded[u]
        ranking.rows = rows
        ranking.expanded = set(kept)
        ranking.lexical_max = max((r.lexical for r in rows.values()), default=0.0)
        ranking.breakdowns = {uid: self._score(ranking, row) for uid, row in rows.items()}

        scores = {uid: b.score for uid, b in ranking.breakdowns.items() if b.score > 0}
        ranking.picks = mmr(scores, lambda a, b: self._sim(ranking, a, b), mu=rp.mmr_mu)
        return ranking

    def _load(self, ranking: _Ranking, ids: list[str], filters: Filters) -> dict[str, CandidateRow]:
        s = ranking.query_vecs.get(VectorView.SUMMARY)
        c = ranking.query_vecs.get(VectorView.CODE)
        return search.load_candidates(
            self.conn,
            ids,
            summary_model=s[0] if s else None,
            code_model=c[0] if c else None,
            tsq=ranking.tsq,
            filters=filters,
        )

    def _score(self, ranking: _Ranking, row: CandidateRow) -> ScoreBreakdown:
        s = ranking.query_vecs.get(VectorView.SUMMARY)
        c = ranking.query_vecs.get(VectorView.CODE)
        lex = row.lexical / ranking.lexical_max if ranking.lexical_max > 0 else 0.0
        pr = ranking.pagerank.get(row.unit.id, 0.0)
        g = pr / ranking.pagerank_max if ranking.pagerank_max > 0 else 0.0
        return score(
            row.unit,
            semantic=cosine(s[1], row.summary_vec) if s else None,
            code=cosine(c[1], row.code_vec) if c else None,
            lexical=min(lex, 1.0),
            graph=g,
            now=ranking.now,
            params=self.settings.retrieval,
            paging=self.settings.paging,
        )

    @staticmethod
    def _sim(ranking: _Ranking, a: str, b: str) -> float:
        """cos(s_a, s_b), falling back to the code view when either lacks a summary vector."""
        ra, rb = ranking.rows[a], ranking.rows[b]
        sim = cosine(ra.summary_vec, rb.summary_vec)
        if sim is None:
            sim = cosine(ra.code_vec, rb.code_vec)
        return max(sim or 0.0, 0.0)

    # Stage 4.

    def _pack(
        self, ranking: _Ranking, budget: int, k: int | None
    ) -> tuple[dict[str, pack.Option], list[str]]:
        picks = ranking.picks[:k] if k is not None else ranking.picks
        candidates = {
            p.unit_id: pack.options(ranking.rows[p.unit_id].unit, max(p.value, 0.0)) for p in picks
        }
        return pack.pack(candidates, budget), [p.unit_id for p in picks]
