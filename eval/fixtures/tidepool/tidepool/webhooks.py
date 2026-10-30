"""Outgoing webhooks for dead-lettered jobs.

When ``webhook_url`` is set, every ``job.dead`` event POSTs a JSON body to it. Each
request is signed so the receiver can check it came from us and is fresh:

    X-Tidepool-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256>

The HMAC key is ``webhook_secret`` and the signed message is the timestamp, a dot,
and the raw body bytes (``b"1700000000." + body``). ``verify`` rejects signatures
more than ``TOLERANCE_SECONDS`` (5 minutes) away from the receiver's clock, which
stops replays of captured requests.

Delivery is attempted up to four times, waiting 1, 5 and 25 seconds between tries
(``DELIVERY_BACKOFF``), on a background thread so the worker never blocks on a slow
receiver. After the last failure the event is dropped and logged; webhooks are a
notification, not a source of truth.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.request
from typing import Any

from tidepool import hooks
from tidepool.config import Settings
from tidepool.job import Job

log = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Tidepool-Signature"
TOLERANCE_SECONDS = 300
DELIVERY_BACKOFF = (1, 5, 25)
TIMEOUT_SECONDS = 10


def sign(secret: str, body: bytes, timestamp: int) -> str:
    mac = hmac.new(secret.encode(), str(timestamp).encode() + b"." + body, hashlib.sha256)
    return f"t={timestamp},v1={mac.hexdigest()}"


def verify(secret: str, body: bytes, header: str, now: float | None = None) -> bool:
    """For receivers: check a signature header against the raw request body."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        timestamp = int(parts["t"])
        received = parts["v1"]
    except (KeyError, ValueError):
        return False
    now = time.time() if now is None else now
    if abs(now - timestamp) > TOLERANCE_SECONDS:
        return False
    expected = sign(secret, body, timestamp).split("v1=", 1)[1]
    return hmac.compare_digest(expected, received)


def job_body(event: str, job: Job) -> bytes:
    data: dict[str, Any] = {
        "event": event,
        "job": {
            "id": job.id,
            "task": job.task,
            "queue": job.queue,
            "attempts": job.attempts,
            "last_error": job.last_error,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        },
    }
    return json.dumps(data, separators=(",", ":")).encode()


def deliver(url: str, secret: str, body: bytes) -> bool:
    for attempt, wait in enumerate((0, *DELIVERY_BACKOFF)):
        if wait:
            time.sleep(wait)
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                SIGNATURE_HEADER: sign(secret, body, int(time.time())),
                "User-Agent": "tidepool-webhooks/1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                if 200 <= resp.status < 300:
                    return True
                log.warning("webhook %s returned %s (try %d)", url, resp.status, attempt + 1)
        except OSError as e:
            log.warning("webhook %s failed: %s (try %d)", url, e, attempt + 1)
    log.error("webhook %s gave up after %d tries", url, len(DELIVERY_BACKOFF) + 1)
    return False


def install(settings: Settings) -> None:
    """Subscribe to ``job.dead`` if a webhook is configured. Called by the worker."""
    if not settings.webhook_url or not settings.webhook_secret:
        return
    url, secret = settings.webhook_url, settings.webhook_secret

    def on_dead(event: str, job: Job) -> None:
        body = job_body(event, job)
        threading.Thread(target=deliver, args=(url, secret, body), daemon=True).start()

    hooks.subscribe("job.dead", on_dead)
