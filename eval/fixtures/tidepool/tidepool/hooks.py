"""In-process event hooks.

Events: ``job.enqueued``, ``job.leased``, ``job.succeeded``, ``job.retrying``,
``job.dead``, ``job.requeued``. Subscribers run synchronously in the thread that
emitted the event; an exception in a subscriber is logged and swallowed so a broken
plugin cannot fail a job.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

Handler = Callable[[str, Any], None]
_SUBSCRIBERS: dict[str, list[Handler]] = defaultdict(list)


def on(event: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        _SUBSCRIBERS[event].append(fn)
        return fn

    return register


def subscribe(event: str, fn: Handler) -> None:
    _SUBSCRIBERS[event].append(fn)


def emit(event: str, subject: Any) -> None:
    for fn in [*_SUBSCRIBERS.get(event, ()), *_SUBSCRIBERS.get("*", ())]:
        try:
            fn(event, subject)
        except Exception:
            log.exception("hook %r failed on %s", getattr(fn, "__name__", fn), event)


def clear() -> None:
    _SUBSCRIBERS.clear()
