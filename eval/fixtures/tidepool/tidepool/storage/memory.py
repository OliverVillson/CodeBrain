"""In-process storage for tests and ``database = ":memory:"``. Not shared between processes."""

from __future__ import annotations

import copy
import threading
from collections import Counter
from collections.abc import Iterable
from datetime import datetime

from tidepool.job import Job, JobState


class MemoryStorage:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._payloads: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def insert(self, job: Job, payload: bytes) -> None:
        with self._lock:
            if job.id in self._jobs:
                raise KeyError(f"duplicate job id {job.id}")
            self._jobs[job.id] = copy.deepcopy(job)
            self._payloads[job.id] = payload

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def payload(self, job_id: str) -> bytes:
        with self._lock:
            return self._payloads[job_id]

    def find_by_dedupe_key(self, queue: str, key: str, since: datetime) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if (
                    job.queue == queue
                    and job.dedupe_key == key
                    and job.enqueued_at is not None
                    and job.enqueued_at >= since
                    and job.state not in (JobState.DEAD, JobState.CANCELLED)
                ):
                    return copy.deepcopy(job)
        return None

    def runnable(self, queue: str, now: datetime, limit: int) -> list[Job]:
        with self._lock:
            out = [
                copy.deepcopy(j)
                for j in self._jobs.values()
                if j.queue == queue
                and j.state is JobState.QUEUED
                and (j.run_at is None or j.run_at <= now)
            ]
        return out[:limit]

    def lease(self, job_id: str, worker: str, now: datetime, expires_at: datetime) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state is not JobState.QUEUED:
                return None
            job.transition(JobState.LEASED)
            job.leased_by = worker
            job.lease_expires_at = expires_at
            return copy.deepcopy(job)

    def renew(self, job_id: str, worker: str, expires_at: datetime) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state is not JobState.LEASED or job.leased_by != worker:
                return False
            job.lease_expires_at = expires_at
            return True

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = copy.deepcopy(job)

    def expired_leases(self, now: datetime) -> list[Job]:
        with self._lock:
            return [
                copy.deepcopy(j)
                for j in self._jobs.values()
                if j.state is JobState.LEASED
                and j.lease_expires_at is not None
                and j.lease_expires_at < now
            ]

    def by_state(self, queue: str | None, state: JobState, limit: int = 100) -> list[Job]:
        with self._lock:
            jobs = [
                copy.deepcopy(j)
                for j in self._jobs.values()
                if j.state is state and (queue is None or j.queue == queue)
            ]
        return jobs[:limit]

    def delete(self, job_ids: Iterable[str]) -> int:
        n = 0
        with self._lock:
            for job_id in job_ids:
                if self._jobs.pop(job_id, None) is not None:
                    self._payloads.pop(job_id, None)
                    n += 1
        return n

    def counts(self) -> dict[tuple[str, JobState], int]:
        with self._lock:
            return dict(Counter((j.queue, j.state) for j in self._jobs.values()))
