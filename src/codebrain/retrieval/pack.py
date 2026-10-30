"""Stage 4: pack units into a token budget at one resolution each (sketch §7).

maximize Σ_u value(u) · ρ_r(u)   subject to   Σ_u t_r(u) ≤ B

This is a multiple-choice knapsack. ``pack`` uses the standard greedy on the
LP relaxation: per unit, keep the options on the upper convex hull of
(tokens, value), turn them into upgrade steps (signature -> summary -> full),
and take steps in order of value gained per token while they fit.

``pack_depth_first`` first gives the leading candidates their full text, in order,
and runs the knapsack on the rest of the budget. ``plan`` is what recall uses: the
same depth-first phase, capped at a share of the budget and aware that a unit inside
another unit's full text is already shown, then the knapsack, then one-line
signatures of what is still left out, as a table of contents.

``render_grouped`` lays the packed units out by file, one heading per file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby

from codebrain.ingest.tokens import approx_tokens
from codebrain.models import VALIDITY_CURRENT, Resolution, Unit


def estimate_tokens(text: str) -> int:
    """Ingest's token approximation, so the packer prices every resolution on one scale."""
    return approx_tokens(text)


def citation(unit: Unit) -> str:
    """``path:start-end`` for code, the unit id for experience units."""
    if unit.path:
        if unit.span:
            return f"{unit.path}:{unit.span.start_line}-{unit.span.end_line}"
        return unit.path
    return unit.id


def _stale(unit: Unit) -> str:
    if unit.labels.get("tombstoned"):
        return " (deleted from the repo)"
    if unit.validity < VALIDITY_CURRENT:
        return " (changed since written)"
    return ""


def header(unit: Unit) -> str:
    name = f" {unit.qualified_name}" if unit.qualified_name else ""
    # Code shows what it is in its own text; experience units say what they are.
    kind = "" if unit.kind.is_code else f" [{unit.kind}]"
    return f"### {citation(unit)}{name}{kind}{_stale(unit)}"


def _collapse_elisions(text: str) -> str:
    """One ``...`` line for a run of them (skeletons elide each child separately)."""
    out: list[str] = []
    for line in text.split("\n"):
        if line.strip() == "..." and out and out[-1].strip() == "...":
            continue
        out.append(line)
    return "\n".join(out)


def body(unit: Unit, resolution: Resolution) -> str | None:
    """The unit's text at ``resolution``, or ``None`` if it has nothing at that level."""
    if resolution is Resolution.SIGNATURE:
        return unit.signature or None
    if resolution is Resolution.SUMMARY:
        return unit.summary or None
    return _collapse_elisions(unit.text)


def render(unit: Unit, resolution: Resolution) -> str:
    text = body(unit, resolution)
    if text is None:
        raise ValueError(f"{unit.id} has no {resolution} text")
    return f"{header(unit)}\n{text}"


def tokens_at(unit: Unit, resolution: Resolution) -> int | None:
    """Tokens of the rendered block. Every resolution uses the same estimator: mixing
    ``unit.tokens`` (whatever counter wrote the unit) for full text with an estimate for
    summaries made a short unit's summary look dearer than its code, and pruned it."""
    if body(unit, resolution) is None:
        return None
    return estimate_tokens(render(unit, resolution))


@dataclass(frozen=True)
class Option:
    resolution: Resolution
    tokens: int
    value: float


def options(
    unit: Unit, value: float, factors: Mapping[Resolution, float] | None = None
) -> list[Option]:
    """Every resolution the unit can be shown at, with ρ_r-scaled value.

    ``factors`` overrides ``Resolution.value_factor`` for some resolutions."""
    out: list[Option] = []
    for r in Resolution:
        t = tokens_at(unit, r)
        if t is not None:
            rho = factors.get(r, r.value_factor) if factors else r.value_factor
            out.append(Option(r, t, value * rho))
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


_PARTIAL_TEXT = ("skeleton", "truncated", "split", "body")
"""Labels of units whose text does not include their children's text."""


def contains(outer: Unit, inner: Unit) -> bool:
    """``inner``'s code is part of ``outer``'s full text (a method inside its class)."""
    if (
        outer.id == inner.id
        or not outer.kind.is_code
        or outer.path is None
        or outer.path != inner.path
        or outer.span is None
        or inner.span is None
        or any(outer.labels.get(k) for k in _PARTIAL_TEXT)
    ):
        return False
    return (
        outer.span.start_line <= inner.span.start_line
        and inner.span.end_line <= outer.span.end_line
        and (outer.span != inner.span or outer.tokens > inner.tokens)
    )


def plan(
    picks: Sequence[tuple[Unit, float]],
    budget: int,
    *,
    full_first: int,
    full_share: float = 1.0,
    toc_share: float = 0.0,
    factors: Mapping[Resolution, float] | None = None,
) -> dict[str, Option]:
    """Pack ranked ``(unit, value)`` picks into ``budget`` tokens.

    1. Depth first: each of the first ``full_first`` picks gets its full text while the
       full texts fit in ``full_share`` of the budget. A pick inside an already chosen
       full text is skipped (it is shown already); a pick whose full text holds already
       chosen picks replaces them.
    2. The knapsack spends the rest, less ``toc_share`` of the budget.
    3. Picks still left out get their signature, in rank order, while it fits: a table
       of contents of what else exists.
    """
    units = {u.id: u for u, _ in picks}
    opts = {u.id: options(u, v, factors) for u, v in picks}
    chosen: dict[str, Option] = {}
    covered: set[str] = set()
    used = 0
    cap = int(budget * full_share)
    tried = 0
    for u, _ in picks:
        if tried >= full_first:
            break
        if u.id in covered:
            continue
        tried += 1
        full = next((o for o in opts[u.id] if o.resolution is Resolution.FULL), None)
        if full is None:
            continue
        inside = [c for c in chosen if contains(u, units[c])]
        refund = sum(chosen[c].tokens for c in inside)
        if used - refund + full.tokens > cap:
            continue
        for c in inside:
            del chosen[c]
        used += full.tokens - refund
        chosen[u.id] = full
        covered |= {o for o in units if contains(u, units[o])}

    reserve = int(budget * toc_share)
    rest = {uid: o for uid, o in opts.items() if uid not in chosen and uid not in covered}
    chosen.update(pack(rest, max(0, budget - used - reserve)))
    used = sum(o.tokens for o in chosen.values())

    for u, _ in picks:
        if u.id in chosen or u.id in covered:
            continue
        sig = next((o for o in opts[u.id] if o.resolution is Resolution.SIGNATURE), None)
        if sig is not None and used + sig.tokens <= budget:
            chosen[u.id] = sig
            used += sig.tokens
    return chosen


def _lines(unit: Unit) -> str:
    if unit.span is None:
        return ""
    s, e = unit.span.start_line, unit.span.end_line
    return f"{s}" if s == e else f"{s}-{e}"


def render_grouped(blocks: Sequence[tuple[Unit, Resolution]]) -> str:
    """The packed units as one context block, grouped by file.

    Files come in the order of their best-ranked unit (``blocks`` is in rank order) and
    each file's units in line order under one ``### path`` heading. Signatures are one
    line each. Units without a path (episodes, decisions) keep their own block.
    """
    first: dict[str, int] = {}
    for i, (u, _) in enumerate(blocks):
        first.setdefault(u.path or u.id, i)

    def file_key(b: tuple[Unit, Resolution]) -> tuple[int, int]:
        u = b[0]
        return first[u.path or u.id], u.span.start_line if u.span else 0

    out: list[str] = []
    for _, group in groupby(sorted(blocks, key=file_key), key=lambda b: file_key(b)[0]):
        items = list(group)
        unit0 = items[0][0]
        if unit0.path is None:
            out.extend(render(u, r) for u, r in items)
            continue
        parts = [f"### {unit0.path}"]
        for u, r in items:
            name = f" {u.qualified_name}" if u.qualified_name and u.qualified_name != u.path else ""
            if r is Resolution.SIGNATURE:
                parts.append(f"- {_lines(u)}: {u.signature}{_stale(u)}")
                continue
            tag = " · summary" if r is Resolution.SUMMARY else ""
            parts.append(f"#### {_lines(u)}{name}{tag}{_stale(u)}\n{body(u, r)}")
        out.append("\n".join(parts))
    return "\n\n".join(out)
