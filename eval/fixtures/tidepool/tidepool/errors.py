"""Exceptions shared across tidepool.

Handlers raise ``PermanentError`` to skip the remaining retries and ``RetryAfter`` to
ask for a specific delay. Everything else a handler raises is treated as transient.
"""

from __future__ import annotations


class TidepoolError(Exception):
    """Base class for errors raised by tidepool itself."""


class ConfigError(TidepoolError):
    """A setting is missing, malformed or out of range."""


class TaskNotFound(TidepoolError):
    """A job names a task that is not registered in this process."""

    def __init__(self, name: str) -> None:
        super().__init__(f"no task registered as {name!r}")
        self.name = name


class LeaseLost(TidepoolError):
    """The worker no longer holds the lease on a job (it expired or was taken)."""


class JobTimeout(TidepoolError):
    """The handler ran past its task's ``timeout``."""


class PayloadError(TidepoolError):
    """A payload could not be encoded or decoded."""


class PermanentError(Exception):
    """Raise from a handler when retrying cannot help, for example a 404 from an API.

    The job goes to the dead-letter queue straight away, whatever attempts remain.
    """


class RetryAfter(Exception):
    """Raise from a handler to retry after ``seconds`` instead of the backoff delay.

    Useful for upstream 429 responses that carry a ``Retry-After`` header. The
    attempt does not count towards ``max_attempts``.
    """

    def __init__(self, seconds: float, reason: str = "") -> None:
        super().__init__(reason or f"retry after {seconds}s")
        self.seconds = seconds
