"""The dead-letter queue: jobs that will not be retried automatically.

A job lands here when it uses up its attempts, raises ``PermanentError`` (or another
never-retry failure), or loses its lease on its final attempt. Dead jobs stay in
storage with state ``dead`` so an operator can inspect and requeue them; they are
purged ``dlq_retention_days`` after they died (14 days by default) by
``purge_expired``, which the CLI's ``tidepool dlq purge`` and the worker's daily
housekeeping both call.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import TYPE_CHECKING

from tidepool import hooks
from tidepool.job import Job, JobState
from tidepool.metrics import METRICS

if TYPE_CHECKING:
    from tidepool.queue import Queue


def move_to_dead(queue: Queue, job: Job, error: str) -> None:
    job.transition(JobState.DEAD)
    job.last_error = error
    job.finished_at = queue.clock.now()
    job.leased_by = job.lease_expires_at = None
    queue.storage.save(job)
    METRICS.inc("tidepool_jobs_total", queue=job.queue, outcome="dead")
    hooks.emit("job.dead", job)


def list_dead(queue: Queue, name: str | None = None, limit: int = 100) -> list[Job]:
    return queue.storage.by_state(name, JobState.DEAD, limit)


def requeue(queue: Queue, job_ids: Iterable[str]) -> list[str]:
    """Give dead jobs a fresh start: attempts back to 0, error and backoff cleared,
    runnable now. Priority and dedupe key are kept. Returns the ids requeued."""
    done = []
    now = queue.clock.now()
    for job_id in job_ids:
        job = queue.storage.get(job_id)
        if job is None or job.state is not JobState.DEAD:
            continue
        job.transition(JobState.QUEUED)
        job.attempts = 0
        job.last_error = None
        job.last_delay = None
        job.finished_at = None
        job.run_at = now
        queue.storage.save(job)
        hooks.emit("job.requeued", job)
        done.append(job_id)
    return done


def requeue_all(queue: Queue, name: str) -> list[str]:
    return requeue(queue, [j.id for j in list_dead(queue, name, limit=10_000)])


def purge_expired(queue: Queue) -> int:
    cutoff = queue.clock.now() - timedelta(days=queue.settings.dlq_retention_days)
    old = [
        j.id
        for j in queue.storage.by_state(None, JobState.DEAD, limit=100_000)
        if j.finished_at is not None and j.finished_at < cutoff
    ]
    n = queue.storage.delete(old)
    METRICS.inc("tidepool_dlq_purged_total", value=n)
    return n
