"""Stage 4: pack units into a token budget at one resolution each (sketch §7).

maximize Σ_u value(u) · ρ_r(u)   subject to   Σ_u t_r(u) ≤ B

This is a multiple-choice knapsack. ``pack`` uses the standard greedy on the
LP relaxation: per unit, keep the options on the upper convex hull of
(tokens, value), turn them into upgrade steps (signature -> summary -> full),
and take steps in order of value gained per token while they fit.

``pack_depth_first`` first gives the leading candidates their full text, in order,
and runs the knapsack on the rest of the budget.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from codebrain.models import VALIDITY_CURRENT, Resolution, Unit


def estimate_tokens(text: str) -> int:
    """About 4 characters per token. Used where ingest has not counted tokens."""
    return math.ceil(len(text) / 4) if text else 0


def citation(unit: Unit) -> str:
    """``path:start-end`` for code, the unit id for experience units."""
    if unit.path:
        if unit.span:
            return f"{unit.path}:{unit.span.start_line}-{unit.span.end_line}"
        return unit.path
    return unit.id


def header(unit: Unit) -> str:
    name = f" {unit.qualified_name}" if unit.qualified_name else ""
    if unit.labels.get("tombstoned"):
        stale = " (deleted from the repo)"
    elif unit.validity < VALIDITY_CURRENT:
        stale = " (changed since written)"
    else:
        stale = ""
    return f"### {citation(unit)}{name} [{unit.kind}]{stale}"


def body(unit: Unit, resolution: Resolution) -> str | None:
    """The unit's text at ``resolution``, or ``None`` if it has nothing at that level."""
    if resolution is Resolution.SIGNATURE:
        return unit.signature or None
    if resolution is Resolution.SUMMARY:
        return unit.summary or None
    return unit.text


def render(unit: Unit, resolution: Resolution) -> str:
    text = body(unit, resolution)
    if text is None:
        raise ValueError(f"{unit.id} has no {resolution} text")
    return f"{header(unit)}\n{text}"


def tokens_at(unit: Unit, resolution: Resolution) -> int | None:
    text = body(unit, resolution)
    if text is None:
        return None
    if resolution is Resolution.FULL and unit.tokens > 0:
        return unit.tokens + estimate_tokens(header(unit)) + 1
    return estimate_tokens(render(unit, resolution))


@dataclass(frozen=True)
class Option:
    resolution: Resolution
    tokens: int
    value: float


def options(unit: Unit, value: float) -> list[Option]:
    """Every resolution the unit can be shown at, with ρ_r-scaled value."""
    out: list[Option] = []
    for r in Resolution:
        t = tokens_at(unit, r)
        if t is not None:
            out.append(Option(r, t, value * r.value_factor))
    return out


def _hull(opts: Sequence[Option]) -> list[Option]:
    """Options on the upper convex hull of (tokens, value), cheapest first."""
    by_cost = sorted(opts, key=lambda o: (o.tokens, -o.value))
    pruned: list[Option] = []
    for o in by_cost:
        if o.value <= 0 or (pruned and o.value <= pruned[-1].value):
            continue  # dominated: costs at least as much for no more value
        pruned.append(o)
    hull: list[Option] = []
    for o in pruned:
        while len(hull) >= 1:
            prev_t, prev_v = (hull[-2].tokens, hull[-2].value) if len(hull) >= 2 else (0, 0.0)
            # Drop the last point if going straight from prev to o is at least as efficient.
            last = hull[-1]
            if (last.value - prev_v) * (o.tokens - prev_t) <= (o.value - prev_v) * (
                last.tokens - prev_t
            ):
                hull.pop()
            else:
                break
        hull.append(o)
    return hull


def pack(candidates: Mapping[str, Sequence[Option]], budget: int) -> dict[str, Option]:
    """Choose at most one option per unit with total tokens ≤ ``budget``."""
    steps: list[tuple[float, int, str, int]] = []  # (efficiency, order, unit, hull index)
    hulls: dict[str, list[Option]] = {}
    order = 0
    for uid, opts in candidates.items():
        hull = _hull(opts)
        hulls[uid] = hull
        prev_t, prev_v = 0, 0.0
        for i, o in enumerate(hull):
            dt = o.tokens - prev_t
            eff = (o.value - prev_v) / dt if dt > 0 else math.inf
            steps.append((eff, order, uid, i))
            order += 1
            prev_t, prev_v = o.tokens, o.value
    steps.sort(key=lambda s: (-s[0], s[1]))

    chosen: dict[str, int] = {}
    blocked: set[str] = set()
    used = 0
    for _, _, uid, i in steps:
        if uid in blocked or chosen.get(uid, -1) != i - 1:
            continue
        prev = hulls[uid][i - 1].tokens if i > 0 else 0
        extra = hulls[uid][i].tokens - prev
        if used + extra > budget:
            # A unit's later upgrades build on this step, so they are skipped too.
            blocked.add(uid)
            continue
        used += extra
        chosen[uid] = i
    return {uid: hulls[uid][i] for uid, i in chosen.items()}


def pack_depth_first(
    candidates: Mapping[str, Sequence[Option]], budget: int, full_first: int
) -> dict[str, Option]:
    """Give the first ``full_first`` candidates (in mapping order) their full option while
    it fits, then ``pack`` the others into what is left. A leading candidate whose full
    text does not fit goes to the knapsack with the rest."""
    chosen: dict[str, Option] = {}
    used = 0
    for uid, opts in list(candidates.items())[:full_first]:
        full = next((o for o in opts if o.resolution is Resolution.FULL), None)
        if full is not None and used + full.tokens <= budget:
            chosen[uid] = full
            used += full.tokens
    rest = {uid: opts for uid, opts in candidates.items() if uid not in chosen}
    chosen.update(pack(rest, budget - used))
    return chosen
