"""The storage contract every backend implements.

Backends own atomicity: ``lease`` must never hand the same job to two workers, and
``renew`` must fail if the caller no longer holds the lease. Everything else
(backoff, priorities, dead-lettering decisions) lives above this layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from tidepool.job import Job, JobState


class Storage(Protocol):
    def insert(self, job: Job, payload: bytes) -> None: ...

    def get(self, job_id: str) -> Job | None: ...

    def payload(self, job_id: str) -> bytes: ...

    def find_by_dedupe_key(self, queue: str, key: str, since: datetime) -> Job | None:
        """A job with this key enqueued at or after ``since`` that is not dead or cancelled."""
        ...

    def runnable(self, queue: str, now: datetime, limit: int) -> list[Job]:
        """Queued jobs whose ``run_at`` has passed, in no particular order."""
        ...

    def lease(
        self, job_id: str, worker: str, now: datetime, expires_at: datetime
    ) -> Job | None:
        """Atomically move a QUEUED job to LEASED. ``None`` if someone else got it first."""
        ...

    def renew(self, job_id: str, worker: str, expires_at: datetime) -> bool: ...

    def save(self, job: Job) -> None:
        """Write back every mutable field of ``job``."""
        ...

    def expired_leases(self, now: datetime) -> list[Job]: ...

    def by_state(self, queue: str | None, state: JobState, limit: int = 100) -> list[Job]: ...

    def delete(self, job_ids: Iterable[str]) -> int: ...

    def counts(self) -> dict[tuple[str, JobState], int]: ...
