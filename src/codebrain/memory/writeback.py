"""Write-back: record a finished task as an episode unit, plus its decisions (sketch §8).

Figure 4 step by step: distill the record, link the episode to the code units in its
diff, merge it into a near-duplicate episode or insert it, write decision units, and
credit the units the task recalled.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from codebrain.config import WriteBackParams, get_settings
from codebrain.memory.credit import CreditReport, apply_credit
from codebrain.memory.diff import parse_diff
from codebrain.memory.distill import PlainDistiller
from codebrain.memory.interfaces import (
    Distiller,
    Embedder,
    MemoryStore,
    TokenCounter,
    approx_tokens,
    normalize,
)
from codebrain.memory.records import DecisionRecord, DistilledTask, TaskRecord
from codebrain.models import (
    VALIDITY_CURRENT,
    VALIDITY_DEPRECATED,
    Edge,
    EdgeType,
    Episode,
    Namespace,
    Unit,
    UnitKind,
    UnitVector,
    VectorView,
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")


def split_identifier(name: str) -> list[str]:
    """getUserById -> get, user, by, id (sketch §4)."""
    return [p.lower() for p in _CAMEL.split(name) if p]


@dataclass
class WriteBackResult:
    episode_id: str
    merged: bool
    """True when the task was folded into an existing near-duplicate episode."""
    version: int
    linked_unit_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    deprecated_ids: list[str] = field(default_factory=list)
    credit: CreditReport = field(default_factory=CreditReport)
    unknown_unit_ids: list[str] = field(default_factory=list)
    """Touched or used ids the harness passed that are not in the store."""


class EpisodeRecorder:
    """Writes finished tasks to the store. The caller owns the transaction."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        distiller: Distiller | None = None,
        params: WriteBackParams | None = None,
        token_counter: TokenCounter = approx_tokens,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.distiller: Distiller = distiller or PlainDistiller()
        self.params = params or get_settings().write_back
        self.count_tokens = token_counter

    def record(self, task: TaskRecord) -> WriteBackResult:
        distilled = self._distill(task)
        changes = parse_diff(task.diff) if task.diff else []

        touched = list(dict.fromkeys(task.touched_unit_ids))
        if task.repo and changes:
            touched += [u.id for u in self.store.units_changed_by(task.repo, changes)]
        code = self.store.get_units(dict.fromkeys(touched))
        linked = [uid for uid in dict.fromkeys(touched) if uid in code]
        given = list(dict.fromkeys(task.touched_unit_ids + task.used_unit_ids))
        known = self.store.get_units(given)
        unknown = [uid for uid in given if uid not in known]

        episode, merged = self._write_episode(task, distilled, [code[u] for u in linked])
        self.store.insert_episode(
            Episode(
                unit_id=episode.id,
                session_id=task.session_id,
                prompt=task.prompt,
                plan=task.plan,
                diff=task.diff,
                test_results=task.test_results,
                tests_passed=task.tests_passed,
                pitfalls=distilled.pitfalls or task.pitfalls,
                started_at=task.started_at,
                ended_at=task.ended_at,
            )
        )
        # Tests passing in the recorded run is extra evidence for the episode's quality.
        if task.tests_passed is not None:
            self.store.add_usage(episode.id, retrieved=1, used=int(task.tests_passed))
        self.store.add_edges(
            Edge(src_id=uid, dst_id=episode.id, type=EdgeType.PRODUCED_BY) for uid in linked
        )

        result = WriteBackResult(
            episode_id=episode.id,
            merged=merged,
            version=episode.version,
            linked_unit_ids=linked,
            unknown_unit_ids=unknown,
        )
        seen = {d.text.strip().lower() for d in task.decisions}
        decisions = task.decisions + [
            DecisionRecord(text=t) for t in distilled.decisions if t.strip().lower() not in seen
        ]
        for rec in decisions:
            decision_id, deprecated = self._write_decision(
                rec, task.repo, task.session_id, episode.id
            )
            result.decision_ids.append(decision_id)
            result.deprecated_ids += deprecated

        if task.session_id:
            result.credit = apply_credit(
                self.store,
                task.session_id,
                since=task.started_at,
                touched=set(linked),
                cited=set(task.used_unit_ids),
                added_lines=(line for c in changes for line in c.added),
                min_length=self.params.min_reference_length,
            )
        return result

    def record_decision(
        self,
        rec: DecisionRecord,
        *,
        repo: str | None = None,
        session_id: str | None = None,
        episode_id: str | None = None,
    ) -> tuple[str, list[str]]:
        """Write one decision outside a task (an ADR, or ``remember`` with kind decision).

        Returns the decision unit id and the ids it deprecated.
        """
        return self._write_decision(rec, repo, session_id, episode_id)

    def _distill(self, task: TaskRecord) -> DistilledTask:
        if task.summary and task.summary.strip():
            lines = [ln.strip() for ln in task.summary.strip().splitlines() if ln.strip()]
            distilled = DistilledTask(
                title=lines[0], built="\n".join(lines[1:]) or lines[0], pitfalls=task.pitfalls
            )
        else:
            distilled = self.distiller.distill_task(task)
        caps = list(dict.fromkeys(distilled.capabilities + task.capabilities))
        return distilled.model_copy(update={"capabilities": caps})

    # Episodes ----------------------------------------------------------------------

    def _write_episode(
        self, task: TaskRecord, distilled: DistilledTask, code: Sequence[Unit]
    ) -> tuple[Unit, bool]:
        summary = distilled.render()
        text = _render_task(task, distilled, code)
        vec = self._embed(summary)
        terms = _lexical_terms(code, distilled.capabilities)

        namespace = Namespace.SESSION if task.session_only else Namespace.PROJECT
        hit = self.store.nearest(
            UnitKind.EPISODE,
            self.embedder.model,
            vec,
            task.repo,
            namespace=namespace,
            session_id=task.session_id,
        )
        if hit and hit[1] >= self.params.dup_threshold:
            old = hit[0]
            version = old.version + 1
            runs = int(old.labels.get("runs", 1)) + 1
            merged_text = f"{old.text}\n\n--- run {runs} ---\n{text}"
            unit = old.model_copy(
                update={
                    "version": version,
                    "summary": summary,
                    "text": merged_text,
                    "tokens": self.count_tokens(merged_text),
                    "lexical_terms": _merge_terms(old.lexical_terms, terms),
                    "capabilities": list(dict.fromkeys(old.capabilities + distilled.capabilities)),
                    "labels": {**old.labels, "runs": runs, "outcome": str(task.outcome)},
                    # The new run is fresh evidence, even if ingest marked the old one stale.
                    "validity": VALIDITY_CURRENT,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._save(unit, vec, summary)
            return unit, True

        unit = Unit(
            kind=UnitKind.EPISODE,
            namespace=namespace,
            repo=task.repo,
            session_id=task.session_id,
            text=text,
            summary=summary,
            lexical_terms=terms or None,
            tokens=self.count_tokens(text),
            capabilities=distilled.capabilities,
            labels={"title": distilled.title, "outcome": str(task.outcome), "runs": 1},
        )
        self._save(unit, vec, summary)
        return unit, False

    # Decisions ---------------------------------------------------------------------

    def _write_decision(
        self,
        rec: DecisionRecord,
        repo: str | None,
        session_id: str | None,
        episode_id: str | None,
    ) -> tuple[str, list[str]]:
        text = rec.text.strip() + (f"\nWhy: {rec.rationale.strip()}" if rec.rationale else "")
        vec = self._embed(text)
        hit = self.store.nearest(
            UnitKind.DECISION, self.embedder.model, vec, repo, namespace=Namespace.PROJECT
        )
        if hit and hit[1] >= self.params.dup_threshold and hit[0].id not in rec.supersedes:
            old = hit[0]
            labels = dict(old.labels)
            labels["episodes"] = list(
                dict.fromkeys([*labels.get("episodes", []), *([episode_id] if episode_id else [])])
            )
            labels["applies_to"] = list(
                dict.fromkeys([*labels.get("applies_to", []), *rec.applies_to])
            )
            unit = old.model_copy(
                update={
                    "version": old.version + 1,
                    "labels": labels,
                    "validity": VALIDITY_CURRENT,
                    "updated_at": datetime.now(UTC),
                }
            )
            self.store.upsert_unit(unit)
        else:
            unit = Unit(
                kind=UnitKind.DECISION,
                namespace=Namespace.PROJECT,
                repo=repo,
                session_id=session_id,
                text=text,
                summary=text,
                tokens=self.count_tokens(text),
                labels={
                    "rationale": rec.rationale,
                    "applies_to": rec.applies_to,
                    "episodes": [episode_id] if episode_id else [],
                },
            )
            self._save(unit, vec, text)
        if episode_id:
            self.store.add_edges(
                [Edge(src_id=unit.id, dst_id=episode_id, type=EdgeType.PRODUCED_BY)]
            )

        stale = self.store.get_units(rec.supersedes)
        deprecated = [
            uid for uid, u in stale.items() if u.kind is UnitKind.DECISION and uid != unit.id
        ]
        if deprecated:
            self.store.set_validity(deprecated, VALIDITY_DEPRECATED)
        return unit.id, deprecated

    # Helpers -----------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return normalize(self.embedder.embed([text])[0])

    def _save(self, unit: Unit, vec: list[float], embedded: str) -> None:
        self.store.upsert_unit(unit)
        self.store.upsert_vector(
            UnitVector(
                unit_id=unit.id,
                view=VectorView.SUMMARY,
                model=self.embedder.model,
                embedding=vec,
                content_hash=_hash(embedded),
            )
        )


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _lexical_terms(code: Iterable[Unit], capabilities: Iterable[str]) -> str:
    terms: list[str] = []
    for u in code:
        for name in (u.qualified_name, u.path):
            if name:
                terms += split_identifier(name)
    for cap in capabilities:
        terms += split_identifier(cap)
    return " ".join(dict.fromkeys(terms))


def _merge_terms(old: str | None, new: str) -> str | None:
    merged = " ".join(dict.fromkeys((old or "").split() + new.split()))
    return merged or None


def _render_task(task: TaskRecord, distilled: DistilledTask, code: Sequence[Unit]) -> str:
    """The full, unsummarized record that the episode unit keeps as its text."""
    parts = [distilled.render(), f"Task: {task.prompt.strip()}", f"Outcome: {task.outcome}"]
    if task.plan:
        parts.append(f"Plan:\n{task.plan.strip()}")
    if task.tests_passed is not None:
        parts.append(f"Tests: {'passed' if task.tests_passed else 'failed'}")
    if code:
        lines = [f"- {u.qualified_name or u.path} ({u.path})" for u in code]
        parts.append("Changed units:\n" + "\n".join(lines))
    return "\n\n".join(parts)
