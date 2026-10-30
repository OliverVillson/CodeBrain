"""Context paging, the replacement for compaction (sketch §9).

``SessionTracker`` is what an agent harness drives. It mirrors the live context window
(system prompt, pinned facts, units paged back in, recent turns) and counts its tokens.
When the window passes the high-water mark (70% by default) the oldest turns are moved
into the store as ``turn`` units in the session namespace, down to the low-water mark
(50%). Their full text is kept and indexed, so recall can bring any of it back; the
window keeps a short index of what was paged out.

Typical loop::

    tracker = SessionTracker("s1", 200_000, store=store, embedder=embedder, system_prompt=sp)
    tracker.start_task(prompt)
    tracker.prefetch(recall)                 # page relevant memory in
    for each message:
        tracker.add_turn(Role.USER, text)    # pages out on its own when needed
        messages = tracker.context()         # what to send to the model
    tracker.finish_task(recorder, diff=..., tests_passed=...)
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from codebrain.config import PagingParams, get_settings
from codebrain.memory.distill import PlainDistiller
from codebrain.memory.interfaces import (
    Distiller,
    Embedder,
    MemoryStore,
    TokenCounter,
    approx_tokens,
    normalize,
)
from codebrain.memory.records import Role, TaskRecord, Turn
from codebrain.memory.writeback import EpisodeRecorder, WriteBackResult
from codebrain.models import Namespace, Unit, UnitKind, UnitVector, VectorView


class PinKind(StrEnum):
    """Pinned facts are never evicted (sketch §9)."""

    TASK = "task"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    FACT = "fact"


class PinnedFact(BaseModel):
    id: str = Field(default_factory=lambda: "pin_" + uuid.uuid4().hex[:12])
    kind: PinKind
    text: str
    tokens: int = Field(ge=0)


class InjectedUnit(BaseModel):
    """A unit paged back into the window. Evicting it just drops it: it is still stored."""

    unit_id: str
    text: str
    tokens: int = Field(ge=0)
    reason: str = "recall"


class Recalled(BaseModel):
    """What a recall function hands back for prefetch."""

    unit_id: str
    text: str
    tokens: int | None = None


class PagedSpan(BaseModel):
    """A run of turns that now lives in the store as one turn unit."""

    unit_id: str
    first_turn: int
    last_turn: int
    tokens: int
    summary: str


class PageEvent(BaseModel):
    tokens_before: int
    tokens_after: int
    evicted_turns: list[int] = Field(default_factory=list)
    units: list[PagedSpan] = Field(default_factory=list)
    dropped_injections: list[str] = Field(default_factory=list)
    over_target: bool = False
    """Still above low water: pinned facts and the kept recent turns alone are too big."""


class ContextMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    source: Literal["system", "pinned", "paged_index", "memory", "turn"]


class SessionState(BaseModel):
    """Everything a tracker holds, so a stateless harness or server can persist it."""

    session_id: str
    window_tokens: int = Field(gt=0)
    repo: str | None = None
    system_prompt: str = ""
    system_tokens: int = 0
    pinned: list[PinnedFact] = Field(default_factory=list)
    injected: list[InjectedUnit] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    paged: list[PagedSpan] = Field(default_factory=list)
    next_turn: int = 1
    plan: str | None = None
    plan_step: str | None = None
    task_started_at: datetime | None = None


Recall = Callable[[str, int], Iterable[Recalled]]
"""``recall(query, token_budget)``, normally the context broker in ``codebrain.retrieval``."""


class SessionTracker:
    def __init__(
        self,
        session_id: str,
        window_tokens: int,
        *,
        store: MemoryStore,
        embedder: Embedder,
        distiller: Distiller | None = None,
        repo: str | None = None,
        system_prompt: str = "",
        params: PagingParams | None = None,
        token_counter: TokenCounter = approx_tokens,
        state: SessionState | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.distiller: Distiller = distiller or PlainDistiller()
        self.params = params or get_settings().paging
        self.count_tokens = token_counter
        self.state = state or SessionState(
            session_id=session_id,
            window_tokens=window_tokens,
            repo=repo,
            system_prompt=system_prompt,
            system_tokens=token_counter(system_prompt) if system_prompt else 0,
        )

    @classmethod
    def from_state(cls, state: SessionState, **kwargs: Any) -> SessionTracker:
        return cls(state.session_id, state.window_tokens, state=state, **kwargs)

    # Accounting --------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def tokens(self) -> int:
        s = self.state
        return (
            s.system_tokens
            + sum(p.tokens for p in s.pinned)
            + sum(i.tokens for i in s.injected)
            + sum(t.tokens for t in s.turns)
            + self._index_tokens()
        )

    @property
    def usage(self) -> float:
        """tokens(window) / W."""
        return self.tokens / self.state.window_tokens

    @property
    def free_tokens(self) -> int:
        """Room left below the high-water mark."""
        return max(0, int(self.params.high_water * self.state.window_tokens) - self.tokens)

    # Window contents ---------------------------------------------------------------

    def pin(self, text: str, kind: PinKind = PinKind.FACT) -> PinnedFact:
        fact = PinnedFact(kind=kind, text=text, tokens=self.count_tokens(text))
        self.state.pinned.append(fact)
        return fact

    def unpin(self, pin_id: str) -> None:
        self.state.pinned = [p for p in self.state.pinned if p.id != pin_id]

    def start_task(self, prompt: str, plan: str | None = None) -> PinnedFact:
        """Pin the task so it is never evicted, and mark when it started for usage credit."""
        self.state.pinned = [p for p in self.state.pinned if p.kind is not PinKind.TASK]
        self.state.task_started_at = datetime.now(UTC)
        self.state.plan = plan
        self.state.plan_step = None
        return self.pin(prompt, PinKind.TASK)

    def set_plan_step(self, step: str | None) -> None:
        self.state.plan_step = step

    def add_turn(
        self, role: Role | str, content: str, tokens: int | None = None
    ) -> PageEvent | None:
        """Append a turn and page out if the window passed high water."""
        turn = Turn(
            index=self.state.next_turn,
            role=Role(role),
            content=content,
            tokens=tokens if tokens is not None else self.count_tokens(content),
        )
        self.state.next_turn += 1
        self.state.turns.append(turn)
        return self.page()

    def inject(
        self, unit_id: str, text: str, tokens: int | None = None, reason: str = "recall"
    ) -> PageEvent | None:
        """Page a stored unit back into the window. Re-injecting the same unit is a no-op."""
        if any(i.unit_id == unit_id for i in self.state.injected):
            return None
        n = tokens if tokens is not None else self.count_tokens(text)
        self.state.injected.append(
            InjectedUnit(unit_id=unit_id, text=text, tokens=n, reason=reason)
        )
        return self.page()

    def prefetch_query(self) -> str:
        """q = last user message ‖ current plan step (sketch §9)."""
        last_user = next(
            (t.content for t in reversed(self.state.turns) if t.role is Role.USER), None
        )
        if last_user is None:
            last_user = next((p.text for p in self.state.pinned if p.kind is PinKind.TASK), "")
        step = self.state.plan_step
        return f"{last_user}\n{step}" if step else last_user

    def prefetch(self, recall: Recall, budget: int | None = None) -> list[str]:
        """Run ``recall`` on the prefetch query and inject what fits. Returns injected ids."""
        query = self.prefetch_query()
        budget = self.free_tokens if budget is None else budget
        if not query or budget <= 0:
            return []
        added: list[str] = []
        for r in recall(query, budget):
            n = r.tokens if r.tokens is not None else self.count_tokens(r.text)
            if n > budget:
                continue
            if any(i.unit_id == r.unit_id for i in self.state.injected):
                continue
            self.state.injected.append(
                InjectedUnit(unit_id=r.unit_id, text=r.text, tokens=n, reason="prefetch")
            )
            budget -= n
            added.append(r.unit_id)
        self.page()
        return added

    def context(self) -> list[ContextMessage]:
        """The window in order: system, pinned, paged-out index, recalled units, turns."""
        s = self.state
        out: list[ContextMessage] = []
        if s.system_prompt:
            out.append(ContextMessage(role="system", content=s.system_prompt, source="system"))
        if s.pinned:
            body = "\n".join(f"- [{p.kind}] {p.text}" for p in s.pinned)
            out.append(ContextMessage(role="system", content=f"Pinned:\n{body}", source="pinned"))
        index = self._index_text()
        if index:
            out.append(ContextMessage(role="system", content=index, source="paged_index"))
        for i in s.injected:
            out.append(
                ContextMessage(
                    role="system",
                    content=f"Recalled from memory ({i.unit_id}):\n{i.text}",
                    source="memory",
                )
            )
        out += [
            ContextMessage(role=t.role.value, content=t.content, source="turn") for t in s.turns
        ]
        return out

    # Paging ------------------------------------------------------------------------

    def page(self, force: bool = False) -> PageEvent | None:
        """Evict down to low water if over high water (or always, with ``force``)."""
        w = self.state.window_tokens
        before = self.tokens
        if not force and before <= self.params.high_water * w:
            return None
        target = self.params.low_water * w
        event = PageEvent(tokens_before=before, tokens_after=before)

        # Recalled units go first: they are already in the store, so dropping them is free.
        while self.state.injected and self.tokens > target:
            event.dropped_injections.append(self.state.injected.pop(0).unit_id)

        keep = max(0, self.params.keep_recent_turns)
        while self.tokens > target:
            removable = max(0, len(self.state.turns) - keep)
            need = self.tokens - target
            n, freed = 0, 0
            while n < removable and freed < need:
                freed += self.state.turns[n].tokens
                n += 1
            if n == 0:
                break
            spans = self._write_turns(self.state.turns[:n])
            # Only change the window once the turns are safely in the store.
            del self.state.turns[:n]
            self.state.paged += spans
            event.units += spans
            event.evicted_turns += [
                i for sp in spans for i in range(sp.first_turn, sp.last_turn + 1)
            ]

        event.tokens_after = self.tokens
        event.over_target = event.tokens_after > target
        if not (event.units or event.dropped_injections):
            return None if not force else event
        return event

    def _groups(self, turns: Sequence[Turn]) -> list[list[Turn]]:
        cap = self.params.turn_unit_max_tokens
        groups: list[list[Turn]] = []
        size = 0
        for t in turns:
            if groups and size + t.tokens <= cap:
                groups[-1].append(t)
                size += t.tokens
            else:
                groups.append([t])
                size = t.tokens
        return groups

    def _write_turns(self, turns: Sequence[Turn]) -> list[PagedSpan]:
        groups = self._groups(turns)
        summaries = [self.distiller.distill_turns(g) or g[0].render() for g in groups]
        vectors = self.embedder.embed(summaries)
        spans: list[PagedSpan] = []
        for group, summary, vec in zip(groups, summaries, vectors, strict=True):
            unit = Unit(
                kind=UnitKind.TURN,
                namespace=Namespace.SESSION,
                session_id=self.session_id,
                repo=self.state.repo,
                text="\n\n".join(t.render() for t in group),
                summary=summary,
                tokens=sum(t.tokens for t in group),
                labels={
                    "first_turn": group[0].index,
                    "last_turn": group[-1].index,
                    "roles": sorted({str(t.role) for t in group}),
                },
                created_at=group[-1].created_at,
            )
            self.store.upsert_unit(unit)
            self.store.upsert_vector(
                UnitVector(
                    unit_id=unit.id,
                    view=VectorView.SUMMARY,
                    model=self.embedder.model,
                    embedding=normalize(vec),
                )
            )
            spans.append(
                PagedSpan(
                    unit_id=unit.id,
                    first_turn=group[0].index,
                    last_turn=group[-1].index,
                    tokens=unit.tokens,
                    summary=summary,
                )
            )
        return spans

    def _index_text(self) -> str:
        if not self.state.paged:
            return ""
        lines = [
            "Earlier turns were paged out to memory, not summarized away. "
            "Recall them by query or unit id:"
        ]
        for sp in self.state.paged:
            head = sp.summary.splitlines()[0] if sp.summary else ""
            lines.append(f"- turns {sp.first_turn}-{sp.last_turn} ({sp.unit_id}): {head}")
        return "\n".join(lines)

    def _index_tokens(self) -> int:
        text = self._index_text()
        return self.count_tokens(text) if text else 0

    # Write-back --------------------------------------------------------------------

    def finish_task(self, recorder: EpisodeRecorder, **fields: Any) -> WriteBackResult:
        """Record the current task as an episode and unpin it.

        ``fields`` are any other ``TaskRecord`` fields (diff, touched_unit_ids,
        tests_passed, outcome, decisions, ...). Prompt, plan, session and start time
        come from the tracker unless given.
        """
        task_pin = next((p for p in self.state.pinned if p.kind is PinKind.TASK), None)
        defaults: dict[str, Any] = {
            "prompt": task_pin.text if task_pin else self.prefetch_query(),
            "session_id": self.session_id,
            "repo": self.state.repo,
            "plan": self.state.plan,
            "started_at": self.state.task_started_at,
        }
        record = TaskRecord(**{**defaults, **fields})
        result = recorder.record(record)
        if task_pin:
            self.unpin(task_pin.id)
        self.state.task_started_at = None
        self.state.plan = self.state.plan_step = None
        return result
