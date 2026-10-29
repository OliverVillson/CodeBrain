"""Inputs to write-back and paging, as an agent harness hands them over."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now(UTC)


class Outcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


class DecisionRecord(BaseModel):
    """A design decision made during a task, written as its own ``decision`` unit."""

    text: str
    rationale: str | None = None
    applies_to: list[str] = Field(
        default_factory=list, description="Unit ids of the code the decision is about."
    )
    supersedes: list[str] = Field(
        default_factory=list,
        description="Decision unit ids this replaces. They are marked deprecated (v = 0).",
    )


class TaskRecord(BaseModel):
    """Everything known about a finished agent task (sketch §8, Figure 4 "Record")."""

    prompt: str
    session_id: str | None = None
    repo: str | None = None
    plan: str | None = None
    diff: str | None = None
    touched_unit_ids: list[str] = Field(
        default_factory=list,
        description="Units the harness knows were edited, on top of what the diff resolves to.",
    )
    used_unit_ids: list[str] = Field(
        default_factory=list,
        description="Recalled units the agent says actually helped. Each gets usage credit.",
    )
    summary: str | None = Field(
        default=None,
        description="A lesson the agent already distilled. When set, the distiller is skipped.",
    )
    capabilities: list[str] = Field(default_factory=list)
    session_only: bool = Field(
        default=False,
        description="Store in the session namespace (short half-life) instead of project.",
    )
    test_results: dict[str, Any] = Field(default_factory=dict)
    tests_passed: bool | None = None
    outcome: Outcome = Outcome.SUCCEEDED
    pitfalls: list[str] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _session_only_needs_session(self) -> TaskRecord:
        if self.session_only and not self.session_id:
            raise ValueError("session_only needs a session_id")
        return self


class DistilledTask(BaseModel):
    """What was built, how, why and the pitfalls (Figure 4 "Distill")."""

    title: str
    built: str
    how: str | None = None
    why: str | None = None
    pitfalls: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(
        default_factory=list, description="Decisions the distiller found that the harness didn't."
    )
    capabilities: list[str] = Field(default_factory=list)

    def render(self) -> str:
        parts = [self.title, f"Built: {self.built}"]
        if self.how:
            parts.append(f"How: {self.how}")
        if self.why:
            parts.append(f"Why: {self.why}")
        if self.pitfalls:
            parts.append("Pitfalls: " + "; ".join(self.pitfalls))
        return "\n".join(parts)


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Turn(BaseModel):
    """One message in a live context window."""

    index: int = Field(ge=1)
    role: Role
    content: str
    tokens: int = Field(ge=0)
    created_at: datetime = Field(default_factory=_now)

    def render(self) -> str:
        return f"[turn {self.index}] {self.role}: {self.content}"
