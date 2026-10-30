"""The queue: enqueue jobs, lease them to workers, and record how attempts ended.

Workers call ``lease_next``, run the handler, then ``complete`` or ``fail``. While a
handler runs, the worker calls ``renew`` to keep its lease (see ``worker.Heartbeat``).
A lease that runs out because its worker died is handed back by ``reap_expired``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from tidepool import deadletter, hooks, payload
from tidepool.clock import SYSTEM_CLOCK, Clock
from tidepool.config import Settings, load_settings
from tidepool.job import Job, JobState
from tidepool.metrics import METRICS
from tidepool.priority import clamp, lease_order
from tidepool.registry import TaskSpec, spec_for
from tidepool.retry import Outcome, RetryPolicy
from tidepool.storage import Storage, open_storage

LEASE_SCAN_LIMIT = 200
"""How many runnable rows ``lease_next`` looks at to pick by effective priority."""


class Queue:
    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        clock: Clock = SYSTEM_CLOCK,
        rng: random.Random | None = None,
    ) -> None:
        self.storage = storage
        self.settings = settings or Settings()
        self.clock = clock
        self.rng = rng or random.Random()

    @classmethod
    def from_settings(cls, **overrides: Any) -> Queue:
        s = load_settings(**overrides)
        return cls(open_storage(s.database), s)

    # Enqueue

    def enqueue(
        self,
        fn: Callable[..., Any] | str,
        *args: Any,
        delay: float = 0.0,
        priority: int | None = None,
        dedupe_key: str | None = None,
        max_attempts: int | None = None,
        **kwargs: Any,
    ) -> Job:
        """Add a job. With a dedupe key (explicit, or derived for ``@task(dedupe=True)``),
        an identical job enqueued within ``dedupe_window_seconds`` is returned instead."""
        spec = spec_for(fn)
        now = self.clock.now()
        if dedupe_key is None and spec.dedupe:
            dedupe_key = derive_dedupe_key(spec, list(args), kwargs)
        if dedupe_key is not None:
            since = now - timedelta(seconds=self.settings.dedupe_window_seconds)
            existing = self.storage.find_by_dedupe_key(spec.queue, dedupe_key, since)
            if existing is not None:
                METRICS.inc("tidepool_enqueue_deduplicated_total", queue=spec.queue)
                return existing
        job = Job(
            task=spec.name,
            queue=spec.queue,
            args=list(args),
            kwargs=kwargs,
            priority=clamp(
                priority if priority is not None
                else spec.priority if spec.priority is not None
                else self.settings.default_priority
            ),
            max_attempts=max_attempts or spec.max_attempts,
            run_at=now + timedelta(seconds=delay),
            enqueued_at=now,
            dedupe_key=dedupe_key,
        )
        data = payload.encode(job.args, job.kwargs)
        self.storage.insert(job, data)
        METRICS.inc("tidepool_jobs_enqueued_total", queue=job.queue)
        hooks.emit("job.enqueued", job)
        return job

    # Leasing

    def lease_next(self, queue: str, worker: str) -> Job | None:
        now = self.clock.now()
        candidates = lease_order(self.storage.runnable(queue, now, LEASE_SCAN_LIMIT), now)
        expires = now + timedelta(seconds=self.settings.lease_seconds)
        for cand in candidates:
            job = self.storage.lease(cand.id, worker, now, expires)
            if job is not None:
                job.args, job.kwargs = payload.decode(self.storage.payload(job.id))
                METRICS.observe(
                    "tidepool_queue_wait_seconds",
                    (now - (job.run_at or now)).total_seconds(),
                    queue=queue,
                )
                hooks.emit("job.leased", job)
                return job
        return None

    def renew(self, job: Job, worker: str) -> bool:
        expires = self.clock.now() + timedelta(seconds=self.settings.lease_seconds)
        ok = self.storage.renew(job.id, worker, expires)
        if ok:
            job.lease_expires_at = expires
        return ok

    # Outcomes

    def complete(self, job: Job) -> None:
        job.transition(JobState.SUCCEEDED)
        job.finished_at = self.clock.now()
        job.leased_by = job.lease_expires_at = None
        self.storage.save(job)
        METRICS.inc("tidepool_jobs_total", queue=job.queue, outcome="succeeded")
        hooks.emit("job.succeeded", job)

    def fail(self, job: Job, exc: BaseException, spec: TaskSpec | None = None) -> Outcome:
        """Record a failed attempt; retry later or dead-letter. Returns which."""
        policy = RetryPolicy.from_settings(self.settings, spec.max_attempts if spec else None)
        decision = policy.decide(job, exc, self.rng)
        job.last_error = decision.error
        if not decision.counts_attempt:
            job.attempts = max(0, job.attempts - 1)
        if decision.outcome is Outcome.DEAD:
            deadletter.move_to_dead(self, job, decision.error)
            return Outcome.DEAD
        job.transition(JobState.QUEUED)
        job.last_delay = decision.delay
        job.run_at = self.clock.now() + timedelta(seconds=decision.delay)
        job.leased_by = job.lease_expires_at = None
        self.storage.save(job)
        METRICS.inc("tidepool_jobs_total", queue=job.queue, outcome="retried")
        hooks.emit("job.retrying", job)
        return Outcome.RETRY

    def reap_expired(self) -> int:
        """Hand back jobs whose worker stopped heartbeating. The lost run counts as an
        attempt, so a job that kills its worker every time still ends up dead."""
        now = self.clock.now()
        n = 0
        for job in self.storage.expired_leases(now):
            job.attempts += 1
            limit = job.max_attempts or self.settings.max_attempts
            if job.attempts >= limit:
                deadletter.move_to_dead(self, job, "lease expired (worker lost)")
            else:
                job.transition(JobState.QUEUED)
                job.last_error = f"lease expired while held by {job.leased_by}"
                job.run_at = now
                job.leased_by = job.lease_expires_at = None
                self.storage.save(job)
            METRICS.inc("tidepool_leases_reaped_total", queue=job.queue)
            n += 1
        return n

    def cancel(self, job_id: str) -> bool:
        job = self.storage.get(job_id)
        if job is None or job.state is not JobState.QUEUED:
            return False
        job.transition(JobState.CANCELLED)
        job.finished_at = self.clock.now()
        self.storage.save(job)
        return True

    def stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for (queue, state), n in self.storage.counts().items():
            out.setdefault(queue, {})[state.value] = n
        return out


def derive_dedupe_key(spec: TaskSpec, args: list[Any], kwargs: dict[str, Any]) -> str:
    blob = json.dumps([spec.name, args, kwargs], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def utcnow(clock: Clock = SYSTEM_CLOCK) -> datetime:
    return clock.now()
