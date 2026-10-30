"""Time source. Tests swap in ``FrozenClock`` to step time by hand."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta


class Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FrozenClock(Clock):
    """A clock that only moves when told to. ``sleep`` advances it instantly."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2024, 1, 1, tzinfo=UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(max(seconds, 0.0))


SYSTEM_CLOCK = Clock()
