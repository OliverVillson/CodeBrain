"""The job record and its state machine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    DEAD = "dead"
    CANCELLED = "cancelled"


TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.LEASED, JobState.CANCELLED}),
    # A failed attempt goes straight back to QUEUED with a later run_at; there is no
    # separate "failed" state, the last error lives on the job.
    JobState.LEASED: frozenset({JobState.SUCCEEDED, JobState.QUEUED, JobState.DEAD}),
    JobState.DEAD: frozenset({JobState.QUEUED}),  # requeued by an operator
    JobState.SUCCEEDED: frozenset(),
    JobState.CANCELLED: frozenset(),
}


class InvalidTransition(ValueError):
    pass


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:20]


@dataclass
class Job:
    task: str
    queue: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_job_id)
    state: JobState = JobState.QUEUED
    priority: int = 5
    attempts: int = 0
    max_attempts: int | None = None
    """Per-job override; ``None`` means the task's value, then the global setting."""
    run_at: datetime | None = None
    enqueued_at: datetime | None = None
    leased_by: str | None = None
    lease_expires_at: datetime | None = None
    last_error: str | None = None
    last_delay: float | None = None
    """The previous backoff delay, which the next one is drawn relative to."""
    dedupe_key: str | None = None
    finished_at: datetime | None = None

    def transition(self, to: JobState) -> None:
        if to not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.id}: {self.state} -> {to}")
        self.state = to

    @property
    def is_terminal(self) -> bool:
        return self.state in (JobState.SUCCEEDED, JobState.CANCELLED)
