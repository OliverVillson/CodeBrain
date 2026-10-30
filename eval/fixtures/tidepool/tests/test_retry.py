import random

from tidepool.errors import PermanentError, RetryAfter
from tidepool.job import Job
from tidepool.retry import Outcome, RetryPolicy


def test_delays_stay_within_base_and_cap() -> None:
    policy = RetryPolicy(base=2.0, cap=900.0, max_attempts=7)
    rng = random.Random(1)
    prev = None
    for _ in range(50):
        d = policy.next_delay(prev, rng)
        assert 2.0 <= d <= 900.0
        prev = d


def test_permanent_error_dead_letters_immediately() -> None:
    job = Job(task="t", queue="q", attempts=1)
    assert RetryPolicy().decide(job, PermanentError("gone")).outcome is Outcome.DEAD


def test_retry_after_does_not_count() -> None:
    job = Job(task="t", queue="q", attempts=3)
    d = RetryPolicy().decide(job, RetryAfter(30))
    assert d.outcome is Outcome.RETRY and d.delay == 30 and not d.counts_attempt


def test_last_attempt_goes_dead() -> None:
    job = Job(task="t", queue="q", attempts=7)
    assert RetryPolicy().decide(job, ValueError("x")).outcome is Outcome.DEAD
