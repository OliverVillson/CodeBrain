"""Graph proximity g(u) by personalized PageRank over the code graph (sketch §6).

The walk treats edges as undirected: from a hit you want both what it calls and
what calls it (the middleware that uses the rate limiter). Edge weights are the
per-type defaults unless the edge stores its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import psycopg
from psycopg.rows import dict_row

from codebrain.models import Edge


def neighborhood(conn: psycopg.Connection, seeds: Iterable[str], hops: int) -> list[Edge]:
    """Every edge touching a unit within ``hops`` undirected hops of the seeds."""
    seen: set[str] = set(seeds)
    frontier = list(seen)
    edges: dict[tuple[str, str, str], Edge] = {}
    for _ in range(hops):
        if not frontier:
            break
        with conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                "SELECT src_id, dst_id, type, weight FROM edges "
                "WHERE src_id = ANY(%(ids)s) OR dst_id = ANY(%(ids)s)",
                {"ids": frontier},
            ).fetchall()
        nxt: set[str] = set()
        for r in rows:
            e = Edge.model_validate(r)
            edges[(e.src_id, e.dst_id, e.type)] = e
            for node in (e.src_id, e.dst_id):
                if node not in seen:
                    nxt.add(node)
        seen |= nxt
        frontier = sorted(nxt)
    return list(edges.values())


def personalized_pagerank(
    seeds: Mapping[str, float],
    edges: Iterable[Edge],
    *,
    beta: float = 0.5,
    iterations: int = 50,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """r = (1 − β) σ + β Pᵀ r with P = D⁻¹ A, A symmetrized.

    σ is ``seeds`` normalized to sum 1. Mass on nodes with no edges returns to σ.
    Returns r for every node touched by the seeds or the edges.
    """
    edge_list = list(edges)
    nodes = sorted(set(seeds) | {n for e in edge_list for n in (e.src_id, e.dst_id)})
    if not nodes:
        return {}
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    sigma = np.zeros(n)
    for uid, s in seeds.items():
        sigma[index[uid]] = max(s, 0.0)
    total = sigma.sum()
    if total <= 0:
        return dict.fromkeys(nodes, 0.0)
    sigma /= total

    src = np.array([index[e.src_id] for e in edge_list] + [index[e.dst_id] for e in edge_list])
    dst = np.array([index[e.dst_id] for e in edge_list] + [index[e.src_id] for e in edge_list])
    w = np.array([e.effective_weight for e in edge_list] * 2, dtype=float)
    out_weight = np.bincount(src, weights=w, minlength=n) if len(w) else np.zeros(n)
    dangling = out_weight == 0
    # Zero-weight edges are allowed; a node whose edges all weigh 0 is dangling, not NaN.
    trans = (
        np.divide(w, out_weight[src], out=np.zeros_like(w), where=out_weight[src] > 0)
        if len(w)
        else w
    )

    r = sigma.copy()
    for _ in range(iterations):
        flow = np.bincount(dst, weights=r[src] * trans, minlength=n) if len(w) else np.zeros(n)
        nxt = (1 - beta) * sigma + beta * (flow + r[dangling].sum() * sigma)
        delta = float(np.abs(nxt - r).sum())
        r = nxt
        if delta < tolerance:
            break
    return {node: float(r[i]) for node, i in index.items()}
