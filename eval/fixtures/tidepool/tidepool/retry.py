"""What happens after a handler fails: retry later, or give up.

Delays use "decorrelated jitter": each delay is drawn uniformly between the base and
three times the previous delay, then capped. Compared with plain exponential backoff
this spreads a burst of failures out instead of retrying them in lockstep, and the
expected delay still grows geometrically.

    delay_1 = uniform(base, 3 * base)
    delay_n = min(cap, uniform(base, 3 * delay_{n-1}))

With the default ``backoff_base`` of 2 s and ``backoff_cap`` of 900 s (15 minutes), the
seventh attempt is typically a few minutes out.
"""

from __future__ import annotations

import random
import traceback
from dataclasses import dataclass
from enum import StrEnum

from tidepool.config import Settings
from tidepool.errors import JobTimeout, PermanentError, RetryAfter, TaskNotFound
from tidepool.job import Job
from tidepool.payload import PayloadError

JITTER_MULTIPLIER = 3.0

NEVER_RETRY: tuple[type[BaseException], ...] = (PermanentError, TaskNotFound, PayloadError)
"""Failures no retry can fix. The job is dead-lettered on the first one."""


class Outcome(StrEnum):
    RETRY = "retry"
    DEAD = "dead"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    delay: float = 0.0
    counts_attempt: bool = True
    error: str = ""


@dataclass(frozen=True)
class RetryPolicy:
    base: float = 2.0
    cap: float = 900.0
    max_attempts: int = 7

    @classmethod
    def from_settings(cls, s: Settings, task_max_attempts: int | None = None) -> RetryPolicy:
        return cls(
            base=s.backoff_base,
            cap=s.backoff_cap,
            max_attempts=task_max_attempts or s.max_attempts,
        )

    def next_delay(self, previous: float | None, rng: random.Random | None = None) -> float:
        rng = rng or random
        upper = max(self.base, JITTER_MULTIPLIER * (previous or self.base))
        return min(self.cap, rng.uniform(self.base, upper))

    def decide(self, job: Job, exc: BaseException, rng: random.Random | None = None) -> Decision:
        """Called after attempt number ``job.attempts`` (already incremented) failed."""
        error = _describe(exc)
        if isinstance(exc, RetryAfter):
            return Decision(
                Outcome.RETRY, min(self.cap, max(0.0, exc.seconds)), counts_attempt=False, error=error
            )
        if isinstance(exc, NEVER_RETRY):
            return Decision(Outcome.DEAD, error=error)
        limit = job.max_attempts or self.max_attempts
        if job.attempts >= limit:
            return Decision(Outcome.DEAD, error=error)
        if isinstance(exc, JobTimeout):
            # A timeout usually means the work is big, not broken: wait at least the cap/10.
            return Decision(Outcome.RETRY, max(self.cap / 10, self.next_delay(job.last_delay, rng)), error=error)
        return Decision(Outcome.RETRY, self.next_delay(job.last_delay, rng), error=error)


def _describe(exc: BaseException) -> str:
    lines = traceback.format_exception_only(type(exc), exc)
    return "".join(lines).strip()[:2000]
