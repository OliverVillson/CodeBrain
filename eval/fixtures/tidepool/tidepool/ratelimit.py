"""Per-queue rate limits with a token bucket.

A limit is written ``N/s``, ``N/m`` or ``N/h`` (for example ``emails=50/s``), optionally
with an explicit burst: ``emails=50/s:200``. Without one, the bucket holds two
seconds' worth of tokens (``2 * rate per second``, at least 1), so a worker that was
idle can briefly catch up but not flood the upstream.

Workers take a token *before* leasing a job. When the bucket is empty the worker does
not lease; it sleeps for the time until the next token, so rate-limited jobs keep
their place in the queue and are not charged an attempt.

Buckets are per worker process. With N worker processes the effective rate is N times
the configured rate; divide accordingly.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from tidepool.clock import SYSTEM_CLOCK, Clock
from tidepool.errors import ConfigError

_LIMIT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*([smh])\s*(?::\s*(\d+))?\s*$")
_PER = {"s": 1.0, "m": 60.0, "h": 3600.0}
DEFAULT_BURST_SECONDS = 2.0


@dataclass(frozen=True)
class Limit:
    rate: float
    """Tokens per second."""
    burst: float


def parse_limit(text: str) -> Limit:
    m = _LIMIT.match(text)
    if not m:
        raise ConfigError(f"bad rate limit {text!r}; expected like 50/s, 10/m or 50/s:200")
    rate = float(m.group(1)) / _PER[m.group(2)]
    if rate <= 0:
        raise ConfigError(f"rate limit {text!r} must be positive")
    burst = float(m.group(3)) if m.group(3) else max(1.0, DEFAULT_BURST_SECONDS * rate)
    return Limit(rate, burst)


class TokenBucket:
    def __init__(self, limit: Limit, clock: Clock = SYSTEM_CLOCK) -> None:
        self.limit = limit
        self.clock = clock
        self.tokens = limit.burst
        self.updated = clock.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock.monotonic()
        self.tokens = min(self.limit.burst, self.tokens + (now - self.updated) * self.limit.rate)
        self.updated = now

    def try_acquire(self, n: float = 1.0) -> float:
        """Take ``n`` tokens if available and return 0, else return seconds to wait."""
        with self._lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return 0.0
            return (n - self.tokens) / self.limit.rate


class RateLimiter:
    """Buckets for the queues that have a limit; other queues are unlimited."""

    def __init__(self, limits: dict[str, str], clock: Clock = SYSTEM_CLOCK) -> None:
        self.buckets = {q: TokenBucket(parse_limit(t), clock) for q, t in limits.items()}

    def wait_time(self, queue: str) -> float:
        bucket = self.buckets.get(queue)
        return 0.0 if bucket is None else bucket.try_acquire()
