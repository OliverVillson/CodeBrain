"""The context broker: ``recall`` and ``explain`` (sketch §7, Figure 3).

Stages, in order:

1. Search the summary, code and lexical indexes (top ``k_per_index`` each), plus an
   AND-ed lexical query over the rarest query terms and exact matches for the
   identifiers in the query, and fuse them with RRF, keeping ``fused_keep``.
2. Expand with personalized PageRank seeded by the best ``ppr_seeds`` fused hits,
   adding up to ``expand_keep`` graph neighbours no index found.
3. Score every candidate with S(u | q), optionally let a reranker boost the ones it
   picks, and drop candidates scoring under ``min_relative_score`` of the best.
4. Order by MMR, where a unit inside another unit's full text is fully redundant, then
   pack into the token budget: the top picks as full code first, the knapsack for the
   rest, one-line signatures for what is still left out. The packing value is the
   score; MMR only orders.

Intent classification, query rewriting and the hypothetical-answer draft from Figure 3
are not here yet.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import psycopg
from pydantic import BaseModel

from codebrain.config import Settings, get_settings
from codebrain.models import Resolution, Retrieval, Unit, VectorView
from codebrain.retrieval import fusion, graph, pack, search
from codebrain.retrieval.encoders import Encoders, Vector, encode_query
from codebrain.retrieval.mmr import Pick, mmr
from codebrain.retrieval.scoring import ScoreBreakdown, path_match, score
from codebrain.retrieval.search import CandidateRow, Filters, TermStats
from codebrain.store import repo

INDEXES = ("semantic", "code", "lexical", "lexical_rare", "symbol")


class Reranker(Protocol):
    """Picks the candidates that answer a query (an LLM or a cross-encoder)."""

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[str]:
        """``candidates`` are ``(unit id, text)``; return the ids that help, best first."""
        ...


def _same_code(a: Unit, b: Unit) -> bool:
    if a.ast_hash is not None and a.ast_hash == b.ast_hash:
        return True
    return a.kind.is_code and b.kind.is_code and a.text == b.text


class _Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + time.perf_counter() - t0


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
    timings: dict[str, float] = {}
    """Seconds spent in each recall stage."""

    def render(self) -> str:
        """The context block to hand the agent, grouped by file."""
        return pack.render_grouped([(u.unit, u.resolution) for u in self.units])


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
    symbols: dict[str, float] = field(default_factory=dict)
    """Symbol-match score per unit, normalized to the best match."""
    boosts: dict[str, float] = field(default_factory=dict)
    terms: frozenset[str] = frozenset()
    """The query's lexical terms, for matching against paths."""
    timer: _Timer = field(default_factory=_Timer)


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
        reranker: Reranker | None = None,
    ) -> None:
        self.conn = conn
        self.encoders: Encoders = encoders or {}
        self.settings = settings or get_settings()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.reranker = reranker
        self._stats: dict[str | None, tuple[tuple[int, str], TermStats]] = {}

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
        with ranking.timer("pack"):
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
            query=query,
            budget=budget,
            tokens_used=sum(u.tokens for u in units),
            units=units,
            timings=ranking.timer.seconds,
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
        timer = _Timer()
        names = search.identifiers(query)
        with timer("embed"):
            dense = query
            if rp.expand_identifiers and names:
                dense = f"{query}\n{' '.join(search.identifier_words(names))}"
            qvecs = encode_query(self.encoders, dense)
        with timer("term_stats"):
            terms = search.lexical_terms(query)
            stats = self._term_stats(filters.repo)
            common = stats.common(terms, rp.lexical_max_df)
            tsq = search.lexical_query(query, drop=common)
            rare = search.rare_query([t for t in terms if t not in common], stats.df)

        rankings: dict[str, list[str]] = {}
        for name, view in (("semantic", VectorView.SUMMARY), ("code", VectorView.CODE)):
            if view in qvecs:
                model, vec = qvecs[view]
                with timer(f"search_{name}"):
                    rankings[name] = search.vector_hits(
                        self.conn, view, model, vec, filters, rp.k_per_index
                    )
        with timer("search_lexical"):
            rankings["lexical"] = search.lexical_hits(self.conn, tsq, filters, rp.k_per_index)
            if rare is not None and rare != tsq:
                rankings["lexical_rare"] = search.lexical_hits(
                    self.conn, rare, filters, rp.k_per_index
                )
        symbols: dict[str, float] = {}
        if names:
            with timer("search_symbol"):
                hits = search.symbol_hits(self.conn, names, filters, rp.k_per_index)
            rankings["symbol"] = [uid for uid, _ in hits]
            best = max((sc for _, sc in hits), default=0.0)
            symbols = {uid: sc / best for uid, sc in hits} if best > 0 else {}

        fused = dict(list(fusion.rrf(rankings, rp.rrf_k).items())[: rp.fused_keep])
        with timer("graph"):
            seeds = dict(list(fused.items())[: gp.ppr_seeds])
            edges = graph.neighborhood(
                self.conn, seeds, gp.ppr_hops, hub_degree=gp.hub_degree, max_nodes=gp.max_nodes
            )
            pr = graph.personalized_pagerank(
                seeds,
                graph.dampen_hubs(edges, gp.hub_degree),
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
            symbols=symbols,
            terms=frozenset(terms),
            timer=timer,
        )
        with timer("load"):
            rows = self._load(ranking, list(fused), filters)
            # Expansion is capped after filtering so filtered-out neighbours don't use
            # slots, but only a bounded prefix of the neighbourhood is loaded.
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
        with timer("score"):
            ranking.breakdowns = {uid: self._score(ranking, row) for uid, row in rows.items()}
        if self.reranker is not None:
            with timer("rerank"):
                self._rerank(query, ranking)

        with timer("mmr"):
            top = max((b.score for b in ranking.breakdowns.values()), default=0.0)
            floor = max(rp.min_relative_score * top, 0.0)
            scores = {
                uid: b.score
                for uid, b in ranking.breakdowns.items()
                if b.score > 0 and b.score >= floor
            }
            ranking.picks = mmr(scores, lambda a, b: self._sim(ranking, a, b), mu=rp.mmr_mu)
        return ranking

    def _term_stats(self, repo_name: str | None) -> TermStats:
        """Per-repo document frequencies, recomputed only when the repo's units change."""
        version = search.stats_version(self.conn, repo_name)
        cached = self._stats.get(repo_name)
        if cached is None or cached[0] != version:
            cached = (version, search.term_stats(self.conn, repo_name))
            self._stats[repo_name] = cached
        return cached[1]

    def _rerank(self, query: str, ranking: _Ranking) -> None:
        assert self.reranker is not None
        rp = self.settings.retrieval
        top = sorted(ranking.breakdowns, key=lambda u: -ranking.breakdowns[u].score)
        top = top[: rp.cross_encoder_top]
        cands = []
        for uid in top:
            u = ranking.rows[uid].unit
            text = u.summary or u.signature or u.text[:400]
            cands.append((uid, f"{pack.header(u)}\n{text}"))
        picked = [uid for uid in self.reranker.rerank(query, cands) if uid in ranking.rows]
        ranking.boosts = {uid: rp.rerank_boost for uid in picked}
        for uid in picked:
            ranking.breakdowns[uid] = self._score(ranking, ranking.rows[uid])

    def _load(self, ranking: _Ranking, ids: list[str], filters: Filters) -> dict[str, CandidateRow]:
        return search.load_candidates(
            self.conn,
            ids,
            summary=ranking.query_vecs.get(VectorView.SUMMARY),
            code=ranking.query_vecs.get(VectorView.CODE),
            tsq=ranking.tsq,
            filters=filters,
        )

    def _score(self, ranking: _Ranking, row: CandidateRow) -> ScoreBreakdown:
        lex = row.lexical / ranking.lexical_max if ranking.lexical_max > 0 else 0.0
        pr = ranking.pagerank.get(row.unit.id, 0.0)
        g = pr / ranking.pagerank_max if ranking.pagerank_max > 0 else 0.0
        return score(
            row.unit,
            semantic=row.semantic,
            code=row.code,
            lexical=min(lex, 1.0),
            graph=g,
            symbol=ranking.symbols.get(row.unit.id, 0.0),
            path=path_match(row.unit, ranking.terms),
            boost=ranking.boosts.get(row.unit.id, 0.0),
            now=ranking.now,
            params=self.settings.retrieval,
            paging=self.settings.paging,
        )

    def _sim(self, ranking: _Ranking, a: str, b: str) -> float:
        """Structural redundancy: 1 when one unit's full text holds the other or both
        are the same code (a vendored copy), else
        ``mmr_same_file`` for two units of one file and 0 otherwise. Embedding similarity
        made a function, its class, its test and the sibling holding the constant
        "redundant", and those are what a question about code needs together."""
        ua, ub = ranking.rows[a].unit, ranking.rows[b].unit
        if pack.contains(ua, ub) or pack.contains(ub, ua) or _same_code(ua, ub):
            return 1.0
        if ua.path is not None and ua.path == ub.path:
            return self.settings.retrieval.mmr_same_file
        return 0.0

    # Stage 4.

    def _pack(
        self, ranking: _Ranking, budget: int, k: int | None
    ) -> tuple[dict[str, pack.Option], list[str]]:
        rp = self.settings.retrieval
        picks = ranking.picks[:k] if k is not None else ranking.picks
        packed = pack.plan(
            [(ranking.rows[p.unit_id].unit, p.score) for p in picks],
            budget,
            full_first=rp.full_first,
            full_share=rp.full_share,
            toc_share=rp.toc_share,
            factors={
                Resolution.SIGNATURE: rp.value_signature,
                Resolution.SUMMARY: rp.value_summary,
            },
        )
        return packed, [p.unit_id for p in picks]
