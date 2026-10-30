"""Periodic jobs.

A ``Schedule`` enqueues a task every ``every`` seconds, or once a day at a fixed UTC
time (``daily_at="03:30"``). The scheduler runs as its own process
(``tidepool scheduler``) and ticks once a second.

Missed runs (the scheduler was down, or a tick was very late) are never replayed one
by one. On the first tick after the due time:

* if the run is at most ``misfire_grace_seconds`` late (60 s by default), it is
  enqueued once, even if several due times were missed ("coalescing");
* if it is later than that, the missed run is skipped and logged.

Either way the next due time is computed from *now*, not from the missed time, so a
long outage never causes a burst.

Only one scheduler should run per database. ``SchedulerLock`` takes a row-level lease
named ``scheduler`` and a second scheduler waits until it expires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from tidepool.clock import SYSTEM_CLOCK, Clock
from tidepool.queue import Queue

log = logging.getLogger(__name__)

TICK_SECONDS = 1.0


@dataclass
class Schedule:
    task: str
    every: float | None = None
    daily_at: str | None = None
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    next_run: datetime | None = None
    skipped: int = 0
    runs: int = 0

    def __post_init__(self) -> None:
        if (self.every is None) == (self.daily_at is None):
            raise ValueError("give exactly one of every= or daily_at=")
        if self.every is not None and self.every < TICK_SECONDS:
            raise ValueError("every= must be at least one second")

    def following(self, after: datetime) -> datetime:
        if self.every is not None:
            return after + timedelta(seconds=self.every)
        assert self.daily_at is not None
        hh, mm = (int(x) for x in self.daily_at.split(":"))
        candidate = datetime.combine(after.date(), time(hh, mm), tzinfo=after.tzinfo)
        return candidate if candidate > after else candidate + timedelta(days=1)


class Scheduler:
    def __init__(self, queue: Queue, schedules: list[Schedule], clock: Clock = SYSTEM_CLOCK) -> None:
        self.queue = queue
        self.schedules = schedules
        self.clock = clock
        self.grace = timedelta(seconds=queue.settings.misfire_grace_seconds)
        now = clock.now()
        for s in schedules:
            s.next_run = s.next_run or s.following(now)

    def tick(self) -> int:
        """Enqueue whatever is due. Returns how many jobs were enqueued."""
        now = self.clock.now()
        enqueued = 0
        for s in self.schedules:
            assert s.next_run is not None
            if now < s.next_run:
                continue
            lateness = now - s.next_run
            if lateness <= self.grace:
                self.queue.enqueue(s.task, *s.args, **s.kwargs)
                s.runs += 1
                enqueued += 1
            else:
                s.skipped += 1
                log.warning(
                    "skipping %s: %.0fs late, grace is %.0fs",
                    s.task, lateness.total_seconds(), self.grace.total_seconds(),
                )
            s.next_run = s.following(now)
        return enqueued

    def run_forever(self) -> None:
        while True:
            self.tick()
            self.clock.sleep(TICK_SECONDS)
