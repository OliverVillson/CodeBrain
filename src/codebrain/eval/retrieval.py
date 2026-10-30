"""Q1: retrieval quality and latency of the baseline pipeline, one row per configuration.

Configurations follow the ablation table in the plan, as far as the baseline goes:
BM25 on split identifiers, dense on source, dense on LLM summaries, RRF of all of them,
and RRF with a HyDE-expanded query (sketch §4, §7 stage 1). Graph expansion, scoring,
MMR and packing get added here once ``retrieval/`` implements them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from codebrain.eval.backends import LLM, CachedEmbedder, Vectors
from codebrain.eval.lexical import BM25, split_identifiers
from codebrain.eval.metrics import (
    bootstrap_ci,
    budget_recall,
    mrr,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from codebrain.eval.mine import GoldTask, Snapshot
from codebrain.eval.units import CodeUnit

TOP = 100
RRF_K = 60
CODE_CHARS = 4000

SUMMARY_SYSTEM = (
    "You write short summaries of code for a search index. In 2 or 3 plain English "
    "sentences say what the code is for, its inputs and outputs, and any side effects. "
    "Do not include code."
)
HYDE_SYSTEM = (
    "You help search a codebase. Given a task, write a 2 or 3 sentence plain English "
    "summary of an existing function or class that would be most useful for doing it, "
    "as a code index would describe it. Do not include code."
)
PARAPHRASE_SYSTEM = (
    "Rewrite the developer's task description in one or two plain English sentences. "
    "Keep the intent. Remove every code identifier, function, class, file and module "
    "name, and every version number. Reply with the rewritten text only."
)


def code_text(u: CodeUnit) -> str:
    return f"# {u.path}\n{u.text[:CODE_CHARS]}"


def summary_key(u: CodeUnit) -> str:
    return "s:" + u.content_hash


async def summarize_units(llm: LLM, units: Sequence[CodeUnit]) -> tuple[dict[str, str], float]:
    """One LLM summary per distinct unit content. Returns summaries and seconds taken."""
    distinct = {u.content_hash: u for u in units}
    t0 = time.perf_counter()

    async def one(u: CodeUnit) -> tuple[str, str]:
        text = await llm.complete(
            SUMMARY_SYSTEM, f"{u.path} :: {u.qualname}\n\n{u.text[:CODE_CHARS]}", 160
        )
        return u.content_hash, text

    pairs = await asyncio.gather(*(one(u) for u in distinct.values()))
    return dict(pairs), time.perf_counter() - t0


async def paraphrase_queries(llm: LLM, tasks: Sequence[GoldTask]) -> None:
    async def one(t: GoldTask) -> None:
        t.query_paraphrase = await llm.complete(PARAPHRASE_SYSTEM, t.query, 200)

    await asyncio.gather(*(one(t) for t in tasks if t.query_paraphrase is None))


async def hyde(llm: LLM, queries: Sequence[str]) -> list[str]:
    return list(await asyncio.gather(*(llm.complete(HYDE_SYSTEM, q, 160) for q in queries)))


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])[:TOP]


def dense_top(matrix: Vectors, q: Vectors, k: int = TOP) -> list[int]:
    if matrix.shape[0] == 0:
        return []
    sims = matrix @ q
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-sims[idx])]]


@dataclass
class QueryResult:
    task_id: str
    variant: str
    config: str
    ranked: list[str]
    seconds: float


def run_retrieval(
    tasks: Sequence[tuple[GoldTask, Snapshot]],
    emb: CachedEmbedder,
    summaries: dict[str, str] | None,
    hyde_texts: dict[str, str] | None,
    variants: Sequence[str],
) -> list[QueryResult]:
    """Rank every snapshot's units for every task, query variant and configuration."""
    all_units = {u.content_hash: u for _, s in tasks for u in s.units}
    emb.ensure([("c:" + h, code_text(u)) for h, u in all_units.items()])
    if summaries is not None:
        emb.ensure([(summary_key(u), summaries[h]) for h, u in all_units.items()])
    token_cache: dict[str, list[str]] = {}

    def toks(u: CodeUnit) -> list[str]:
        if u.content_hash not in token_cache:
            token_cache[u.content_hash] = split_identifiers(f"{u.path} {u.text}")
        return token_cache[u.content_hash]

    results: list[QueryResult] = []
    for task, snap in tasks:
        units = snap.units
        keys = [u.key for u in units]
        bm25 = BM25([toks(u) for u in units])
        code_m = emb.matrix(["c:" + u.content_hash for u in units])
        sum_m = emb.matrix([summary_key(u) for u in units]) if summaries is not None else None

        for variant in variants:
            q = task.query if variant == "original" else (task.query_paraphrase or task.query)
            qv = emb.query([q])[0]
            hv = None
            if hyde_texts is not None:
                h = hyde_texts.get(f"{task.id}|{variant}")
                if h:
                    hv = emb.query([h])[0]

            timed: dict[str, tuple[list[int], float]] = {}
            t0 = time.perf_counter()
            timed["bm25"] = (bm25.top(split_identifiers(q), TOP), time.perf_counter() - t0)
            t0 = time.perf_counter()
            timed["dense_code"] = (dense_top(code_m, qv), time.perf_counter() - t0)
            if sum_m is not None:
                t0 = time.perf_counter()
                timed["dense_summary"] = (dense_top(sum_m, qv), time.perf_counter() - t0)
            t0 = time.perf_counter()
            fused = rrf([r for r, _ in timed.values()])
            timed["rrf"] = (fused, time.perf_counter() - t0 + sum(s for _, s in timed.values()))
            if hv is not None:
                t0 = time.perf_counter()
                mixed = qv + hv
                mixed = (mixed / (np.linalg.norm(mixed) or 1.0)).astype(np.float32)
                ranks = [timed["bm25"][0], dense_top(code_m, mixed)]
                if sum_m is not None:
                    ranks.append(dense_top(sum_m, mixed))
                timed["rrf_hyde"] = (
                    rrf(ranks),
                    time.perf_counter() - t0 + timed["bm25"][1],
                )
            for name, (ranked, secs) in timed.items():
                results.append(QueryResult(task.id, variant, name, [keys[i] for i in ranked], secs))
    return results


def score(
    results: Sequence[QueryResult],
    tasks: Sequence[tuple[GoldTask, Snapshot]],
    count_tokens: Callable[[str], int],
    budget: int = 8000,
) -> list[dict[str, object]]:
    """Aggregate metrics per (variant, config) with 95% bootstrap intervals."""
    gold = {t.id: t for t, _ in tasks}
    unit_tokens: dict[str, int] = {}
    for _, snap in tasks:
        for u in snap.units:
            if u.key not in unit_tokens:
                unit_tokens[u.key] = count_tokens(u.text)
    groups: dict[tuple[str, str], list[QueryResult]] = {}
    for r in results:
        groups.setdefault((r.variant, r.config), []).append(r)

    rows: list[dict[str, object]] = []
    for (variant, config), rs in groups.items():
        per: dict[str, list[float]] = {"r@10": [], "r@20": [], "mrr": [], "ndcg@10": [], "br": []}
        for r in rs:
            rel = gold[r.task_id].relevant
            per["r@10"].append(recall_at_k(r.ranked, rel, 10))
            per["r@20"].append(recall_at_k(r.ranked, rel, 20))
            per["mrr"].append(mrr(r.ranked, rel))
            per["ndcg@10"].append(ndcg_at_k(r.ranked, rel, 10))
            per["br"].append(budget_recall(r.ranked, rel, lambda k: unit_tokens.get(k, 0), budget))
        row: dict[str, object] = {"variant": variant, "config": config, "n": len(rs)}
        for m, vals in per.items():
            row[m] = sum(vals) / len(vals)
            row[m + "_ci"] = bootstrap_ci(vals)
        secs = [r.seconds * 1000 for r in rs]
        row["p50_ms"] = percentile(secs, 50)
        row["p95_ms"] = percentile(secs, 95)
        rows.append(row)
    order = ["bm25", "dense_code", "dense_summary", "rrf", "rrf_hyde"]
    rows.sort(key=lambda r: (str(r["variant"]), order.index(str(r["config"]))))
    return rows
