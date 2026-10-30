"""Graph proximity g(u) by personalized PageRank over the code graph (sketch §6).

The walk treats edges as undirected: from a hit you want both what it calls and
what calls it (the middleware that uses the rate limiter). Edge weights are the
per-type defaults unless the edge stores its own.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import NamedTuple, Protocol

import numpy as np
import psycopg

from codebrain.models import EdgeType


class WeightedEdge(Protocol):
    @property
    def src_id(self) -> str: ...
    @property
    def dst_id(self) -> str: ...
    @property
    def effective_weight(self) -> float: ...


class GraphEdge(NamedTuple):
    """An edge as the walk needs it: no model validation per row (there are thousands)."""

    src_id: str
    dst_id: str
    type: EdgeType
    weight: float | None = None

    @property
    def effective_weight(self) -> float:
        return self.weight if self.weight is not None else self.type.default_weight


def neighborhood(
    conn: psycopg.Connection,
    seeds: Iterable[str],
    hops: int,
    *,
    hub_degree: int | None = None,
    max_nodes: int | None = None,
) -> list[GraphEdge]:
    """Every edge touching a unit within ``hops`` undirected hops of the seeds.

    After the first hop, units with more than ``hub_degree`` edges are not expanded
    further (a big module reaches every unit in its file and, through imports, half the
    repo), and no further hop is loaded once ``max_nodes`` units are known.
    """
    seen: set[str] = set(seeds)
    frontier = sorted(seen)
    edges: dict[tuple[str, str, str], GraphEdge] = {}
    for hop in range(hops):
        if hop > 0 and max_nodes is not None and len(seen) >= max_nodes:
            break
        if hop > 0 and hub_degree is not None:
            # A unit with no edges is not in ``degrees`` and has nothing to expand anyway.
            deg = degrees(conn, frontier)
            frontier = sorted(n for n, d in deg.items() if d <= hub_degree)
        if not frontier:
            break
        rows = conn.execute(
            "SELECT src_id, dst_id, type::text, weight FROM edges "
            "WHERE src_id = ANY(%(ids)s) OR dst_id = ANY(%(ids)s)",
            {"ids": frontier},
        ).fetchall()
        nxt: set[str] = set()
        for src, dst, typ, weight in rows:
            edges[(src, dst, typ)] = GraphEdge(src, dst, EdgeType(typ), weight)
            for node in (src, dst):
                if node not in seen:
                    nxt.add(node)
        seen |= nxt
        frontier = sorted(nxt)
    return list(edges.values())


def degrees(conn: psycopg.Connection, ids: list[str]) -> dict[str, int]:
    """Undirected degree of each unit in ``ids`` (units with no edges are left out)."""
    rows = conn.execute(
        "SELECT n, count(*) FROM ("
        " SELECT src_id AS n FROM edges WHERE src_id = ANY(%(ids)s)"
        " UNION ALL SELECT dst_id FROM edges WHERE dst_id = ANY(%(ids)s)"
        ") d GROUP BY n",
        {"ids": ids},
    ).fetchall()
    return {str(n): int(c) for n, c in rows}


def dampen_hubs(edges: Iterable[GraphEdge], hub_degree: int) -> list[GraphEdge]:
    """Scale each ``contains`` edge of a container with more than ``hub_degree`` children
    by ``hub_degree / children``, so a walk does not spread evenly over a whole file."""
    edge_list = list(edges)
    fanout = Counter(e.src_id for e in edge_list if e.type is EdgeType.CONTAINS)
    out: list[GraphEdge] = []
    for e in edge_list:
        n = fanout[e.src_id] if e.type is EdgeType.CONTAINS else 0
        if n > hub_degree:
            e = e._replace(weight=e.effective_weight * hub_degree / n)
        out.append(e)
    return out


def personalized_pagerank(
    seeds: Mapping[str, float],
    edges: Iterable[WeightedEdge],
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
