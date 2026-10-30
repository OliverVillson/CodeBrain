"""Usage credit: raise the quality prior of units that a task actually reused (sketch §8).

Recall already counts ``retrieved_u`` when it logs a retrieval. When the task ends,
each open retrieval in the session is judged: ``used_u`` goes up if the unit was
edited, called or imported in the final diff, or named by the agent as having helped. An unused
retrieval only leaves ``retrieved_u`` higher, which is what lowers
Q(u) = (used + 1) / (retrieved + 2).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from codebrain.memory.interfaces import MemoryStore
from codebrain.models import Unit, UnitKind


@dataclass
class CreditReport:
    used: list[str] = field(default_factory=list)
    """Unit ids whose ``used_count`` went up."""
    unused: list[str] = field(default_factory=list)
    """Unit ids that were recalled and not reused."""
    retrievals: int = 0
    """How many logged retrievals were judged."""


def reference_names(unit: Unit) -> list[str]:
    """Identifiers that, appearing in added code, mean the unit was called or imported."""
    names: list[str] = []
    if unit.kind is UnitKind.MODULE:
        # A module's qualified name is usually its path, so never split it on "." blindly:
        # the last piece would be the extension ("java", "tsx"), which matches unrelated code.
        if unit.path:
            dotted = re.sub(r"\.[A-Za-z0-9]+$", "", unit.path).replace("/", ".")
            if dotted.endswith(".__init__"):
                dotted = dotted.removesuffix(".__init__")
            names += [dotted.rsplit(".", 1)[-1], dotted]
        if unit.qualified_name and unit.qualified_name != unit.path:
            parts = re.split(r"[.:/]+", unit.qualified_name)
            names += [parts[-1], ".".join(parts)]
    elif unit.qualified_name:
        names.append(re.split(r"[.:/]+", unit.qualified_name)[-1])
    return list(dict.fromkeys(n for n in names if n))


def is_referenced(unit: Unit, added: str, min_length: int) -> bool:
    if not unit.kind.is_code or not added:
        return False
    for name in reference_names(unit):
        if len(name) >= min_length and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", added):
            return True
        # Dotted module names also count as a suffix, e.g. "from pkg.ratelimit.bucket".
        if "." in name and len(name) >= min_length and name in added:
            return True
    return False


def apply_credit(
    store: MemoryStore,
    session_id: str,
    *,
    since: datetime | None,
    touched: Collection[str],
    cited: Collection[str],
    added_lines: Iterable[str],
    min_length: int = 3,
) -> CreditReport:
    retrievals = [r for r in store.open_retrievals(session_id, since) if r.id is not None]
    if not retrievals:
        return CreditReport()
    units = store.get_units(r.unit_id for r in retrievals)
    added = "\n".join(added_lines)

    verdict: dict[str, bool] = {}
    for uid, unit in units.items():
        verdict[uid] = uid in touched or uid in cited or is_referenced(unit, added, min_length)

    labels: dict[int, bool] = {}
    used_rows: Counter[str] = Counter()
    for r in retrievals:
        assert r.id is not None
        labels[r.id] = verdict.get(r.unit_id, False)
        if labels[r.id]:
            # One credit per retrieval row, so Q(u) stays a ratio of the same events.
            used_rows[r.unit_id] += 1

    store.mark_retrievals(labels)
    for uid, n in used_rows.items():
        store.add_usage(uid, retrieved=0, used=n)

    recalled = list(dict.fromkeys(r.unit_id for r in retrievals))
    return CreditReport(
        used=[u for u in recalled if used_rows[u]],
        unused=[u for u in recalled if not used_rows[u]],
        retrievals=len(retrievals),
    )
