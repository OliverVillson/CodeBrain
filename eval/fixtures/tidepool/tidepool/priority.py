"""Priorities and aging.

Priorities run from 0 (lowest) to 9 (highest); tasks default to 5. A queue that is
always busy with high-priority work would starve everything else, so a job's
*effective* priority rises the longer it waits: one step per ``AGING_STEP_SECONDS``
in the queue, by at most ``MAX_AGING_BOOST`` steps, and never past ``MAX_PRIORITY``.

Leasing picks the highest effective priority first and, among equals, the job that
has been runnable longest.
"""

from __future__ import annotations

from datetime import datetime

from tidepool.job import Job

MIN_PRIORITY = 0
MAX_PRIORITY = 9
AGING_STEP_SECONDS = 90
MAX_AGING_BOOST = 5


def clamp(priority: int) -> int:
    return max(MIN_PRIORITY, min(MAX_PRIORITY, priority))


def aging_boost(waited_seconds: float) -> int:
    if waited_seconds <= 0:
        return 0
    return min(MAX_AGING_BOOST, int(waited_seconds // AGING_STEP_SECONDS))


def effective_priority(job: Job, now: datetime) -> int:
    since = job.run_at or job.enqueued_at or now
    return clamp(job.priority + aging_boost((now - since).total_seconds()))


def lease_order(jobs: list[Job], now: datetime) -> list[Job]:
    """Jobs in the order they should be leased."""
    return sorted(
        jobs,
        key=lambda j: (-effective_priority(j, now), j.run_at or j.enqueued_at or now, j.id),
    )
