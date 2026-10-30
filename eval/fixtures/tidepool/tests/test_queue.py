from tidepool.clock import FrozenClock
from tidepool.config import Settings
from tidepool.job import JobState
from tidepool.queue import Queue
from tidepool.registry import task
from tidepool.storage import MemoryStorage


@task(name="tests.add", queue="math", dedupe=True)
def add(a: int, b: int) -> int:
    return a + b


def make_queue() -> tuple[Queue, FrozenClock]:
    clock = FrozenClock()
    return Queue(MemoryStorage(), Settings(database=":memory:"), clock=clock), clock


def test_dedupe_returns_existing_job() -> None:
    q, _ = make_queue()
    a = q.enqueue(add, 1, 2)
    b = q.enqueue(add, 1, 2)
    assert a.id == b.id


def test_expired_lease_is_reaped_and_counts_an_attempt() -> None:
    q, clock = make_queue()
    job = q.enqueue(add, 2, 3)
    leased = q.lease_next("math", "w1")
    assert leased is not None and leased.id == job.id
    clock.advance(61)
    assert q.reap_expired() == 1
    again = q.storage.get(job.id)
    assert again is not None and again.state is JobState.QUEUED and again.attempts == 1
