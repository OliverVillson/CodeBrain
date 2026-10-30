"""The worker: lease jobs, run handlers, report outcomes.

One ``Worker`` runs ``concurrency`` slots on a thread pool. Each slot loops:

1. ask the rate limiter for a token (sleep instead of leasing if there is none);
2. ``lease_next`` on each of its queues in order;
3. run the handler under a ``Heartbeat`` that renews the lease every
   ``settings.heartbeat_interval`` seconds (``lease_seconds / heartbeat_divisor``, so
   20 s with the defaults);
4. ``complete`` or ``fail`` the job.

If a renewal fails the lease is gone (another worker may already be running the job),
so the slot stops heartbeating and drops the result instead of completing it.

Housekeeping on its own thread calls ``Queue.reap_expired`` every
``reap_interval_seconds`` (15 s) and ``deadletter.purge_expired`` once a day.

On SIGTERM or SIGINT the worker stops leasing, waits up to ``shutdown_grace_seconds``
(30 s) for running handlers, and then exits. Handlers still running at that point are
abandoned; their leases expire and the reaper hands the jobs to another worker.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from types import FrameType

from tidepool import deadletter, webhooks
from tidepool.errors import JobTimeout, LeaseLost
from tidepool.job import Job
from tidepool.metrics import METRICS
from tidepool.queue import Queue
from tidepool.ratelimit import RateLimiter
from tidepool.registry import get_task

log = logging.getLogger(__name__)

PURGE_INTERVAL_SECONDS = 86_400


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident() % 10_000}"


class Heartbeat:
    """Renews a job's lease on a timer while the handler runs."""

    def __init__(self, queue: Queue, job: Job, worker: str) -> None:
        self.queue = queue
        self.job = job
        self.worker = worker
        self.interval = queue.settings.heartbeat_interval
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"hb-{job.id}")

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if not self.queue.renew(self.job, self.worker):
                log.warning("lost lease on %s", self.job.id)
                self.lost.set()
                return


class Worker:
    def __init__(self, queue: Queue, queues: Sequence[str] = ("default",)) -> None:
        self.queue = queue
        self.queues = list(queues)
        self.settings = queue.settings
        self.limiter = RateLimiter(self.settings.rate_limits, queue.clock)
        self.id = worker_id()
        self.stopping = threading.Event()
        self._pool = ThreadPoolExecutor(self.settings.concurrency, thread_name_prefix="slot")
        self._handlers = ThreadPoolExecutor(self.settings.concurrency * 2, thread_name_prefix="h")

    # Lifecycle

    def run(self) -> None:
        webhooks.install(self.settings)
        self._install_signals()
        hk = threading.Thread(target=self._housekeeping, daemon=True, name="housekeeping")
        hk.start()
        slots = [self._pool.submit(self._slot, i) for i in range(self.settings.concurrency)]
        self.stopping.wait()
        log.info("stopping: waiting up to %ss for running jobs", self.settings.shutdown_grace_seconds)
        deadline = time.monotonic() + self.settings.shutdown_grace_seconds
        for f in slots:
            try:
                f.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeout:
                log.warning("abandoning a running job after the grace period")
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._handlers.shutdown(wait=False, cancel_futures=True)

    def stop(self, *_: object) -> None:
        self.stopping.set()

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum: int, frame: FrameType | None) -> None:
            log.info("got signal %s", signum)
            self.stop()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    # Slots

    def _slot(self, index: int) -> None:
        while not self.stopping.is_set():
            job = self._next_job()
            if job is None:
                self.stopping.wait(self.settings.poll_interval)
                continue
            self.run_job(job)

    def _next_job(self) -> Job | None:
        for name in self.queues:
            wait = self.limiter.wait_time(name)
            if wait > 0:
                METRICS.inc("tidepool_rate_limited_seconds_total", value=wait, queue=name)
                self.stopping.wait(wait)
                continue
            job = self.queue.lease_next(name, self.id)
            if job is not None:
                return job
        return None

    def run_job(self, job: Job) -> None:
        try:
            spec = get_task(job.task)
        except Exception as e:
            job.attempts += 1
            self.queue.fail(job, e)
            return
        job.attempts += 1
        started = time.monotonic()
        with Heartbeat(self.queue, job, self.id) as hb:
            future: Future[object] = self._handlers.submit(spec.fn, *job.args, **job.kwargs)
            try:
                future.result(timeout=spec.timeout)
                error: BaseException | None = None
            except FutureTimeout:
                future.cancel()
                error = JobTimeout(f"{spec.name} ran longer than {spec.timeout}s")
            except BaseException as e:  # the handler's own failure
                error = e
        METRICS.observe(
            "tidepool_job_duration_seconds", time.monotonic() - started, queue=job.queue, task=spec.name
        )
        if hb.lost.is_set():
            log.warning("dropping result of %s: %s", job.id, LeaseLost(job.id))
            return
        if error is None:
            self.queue.complete(job)
        else:
            self.queue.fail(job, error, spec)

    # Housekeeping

    def _housekeeping(self) -> None:
        last_purge = 0.0
        while not self.stopping.wait(self.settings.reap_interval_seconds):
            try:
                n = self.queue.reap_expired()
                if n:
                    log.info("reaped %d expired leases", n)
                if time.monotonic() - last_purge > PURGE_INTERVAL_SECONDS:
                    deadletter.purge_expired(self.queue)
                    last_purge = time.monotonic()
            except Exception:
                log.exception("housekeeping failed")
