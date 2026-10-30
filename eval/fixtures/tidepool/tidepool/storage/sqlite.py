"""SQLite storage. One file, many worker processes on the same host.

Leasing uses a conditional UPDATE (``... WHERE state = 'queued'``) inside a
``BEGIN IMMEDIATE`` transaction, so two workers racing for the same row cannot both
win: the loser sees zero rows changed. WAL mode keeps readers (the admin API, stats)
from blocking workers.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from tidepool.job import Job, JobState

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    task             TEXT NOT NULL,
    queue            TEXT NOT NULL,
    state            TEXT NOT NULL,
    priority         INTEGER NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER,
    run_at           TEXT,
    enqueued_at      TEXT,
    leased_by        TEXT,
    lease_expires_at TEXT,
    last_error       TEXT,
    last_delay       REAL,
    dedupe_key       TEXT,
    finished_at      TEXT,
    payload          BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_runnable ON jobs (queue, state, run_at);
CREATE INDEX IF NOT EXISTS jobs_dedupe ON jobs (queue, dedupe_key, enqueued_at)
    WHERE dedupe_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS jobs_leases ON jobs (state, lease_expires_at);
"""

_COLUMNS = (
    "id task queue state priority attempts max_attempts run_at enqueued_at leased_by "
    "lease_expires_at last_error last_delay dedupe_key finished_at"
).split()
_DATES = {"run_at", "enqueued_at", "lease_expires_at", "finished_at"}


def _ts(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _row_to_job(row: sqlite3.Row) -> Job:
    data: dict[str, Any] = {c: row[c] for c in _COLUMNS}
    for c in _DATES:
        data[c] = datetime.fromisoformat(data[c]) if data[c] else None
    data["state"] = JobState(data["state"])
    return Job(args=[], kwargs={}, **data)


class SQLiteStorage:
    def __init__(self, path: str) -> None:
        self.path = path
        self._local = threading.local()
        with self._tx() as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=30000")
            self._local.db = db
        return db

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        db = self._db()
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
        except BaseException:
            db.execute("ROLLBACK")
            raise
        db.execute("COMMIT")

    def insert(self, job: Job, payload: bytes) -> None:
        values = [getattr(job, c) for c in _COLUMNS]
        values = [_ts(v) if isinstance(v, datetime) else v for v in values]
        with self._tx() as db:
            db.execute(
                f"INSERT INTO jobs ({', '.join(_COLUMNS)}, payload) "
                f"VALUES ({', '.join('?' * (len(_COLUMNS) + 1))})",
                [*values, payload],
            )

    def get(self, job_id: str) -> Job | None:
        row = self._db().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def payload(self, job_id: str) -> bytes:
        row = self._db().execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return bytes(row["payload"])

    def find_by_dedupe_key(self, queue: str, key: str, since: datetime) -> Job | None:
        row = self._db().execute(
            "SELECT * FROM jobs WHERE queue = ? AND dedupe_key = ? AND enqueued_at >= ? "
            "AND state NOT IN ('dead', 'cancelled') ORDER BY enqueued_at DESC LIMIT 1",
            (queue, key, _ts(since)),
        ).fetchone()
        return _row_to_job(row) if row else None

    def runnable(self, queue: str, now: datetime, limit: int) -> list[Job]:
        rows = self._db().execute(
            "SELECT * FROM jobs WHERE queue = ? AND state = 'queued' "
            "AND (run_at IS NULL OR run_at <= ?) LIMIT ?",
            (queue, _ts(now), limit),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def lease(self, job_id: str, worker: str, now: datetime, expires_at: datetime) -> Job | None:
        with self._tx() as db:
            cur = db.execute(
                "UPDATE jobs SET state = 'leased', leased_by = ?, lease_expires_at = ? "
                "WHERE id = ? AND state = 'queued'",
                (worker, _ts(expires_at), job_id),
            )
            if cur.rowcount != 1:
                return None
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row)

    def renew(self, job_id: str, worker: str, expires_at: datetime) -> bool:
        with self._tx() as db:
            cur = db.execute(
                "UPDATE jobs SET lease_expires_at = ? "
                "WHERE id = ? AND state = 'leased' AND leased_by = ?",
                (_ts(expires_at), job_id, worker),
            )
            return cur.rowcount == 1

    def save(self, job: Job) -> None:
        cols = [c for c in _COLUMNS if c != "id"]
        values = [getattr(job, c) for c in cols]
        values = [_ts(v) if isinstance(v, datetime) else v for v in values]
        with self._tx() as db:
            db.execute(
                f"UPDATE jobs SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                [*values, job.id],
            )

    def expired_leases(self, now: datetime) -> list[Job]:
        rows = self._db().execute(
            "SELECT * FROM jobs WHERE state = 'leased' AND lease_expires_at < ?", (_ts(now),)
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def by_state(self, queue: str | None, state: JobState, limit: int = 100) -> list[Job]:
        if queue is None:
            rows = self._db().execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY enqueued_at LIMIT ?",
                (state.value, limit),
            ).fetchall()
        else:
            rows = self._db().execute(
                "SELECT * FROM jobs WHERE queue = ? AND state = ? ORDER BY enqueued_at LIMIT ?",
                (queue, state.value, limit),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def delete(self, job_ids: Iterable[str]) -> int:
        ids = list(job_ids)
        if not ids:
            return 0
        with self._tx() as db:
            cur = db.execute(
                f"DELETE FROM jobs WHERE id IN ({', '.join('?' * len(ids))})", ids
            )
            return cur.rowcount

    def counts(self) -> dict[tuple[str, JobState], int]:
        rows = self._db().execute(
            "SELECT queue, state, COUNT(*) AS n FROM jobs GROUP BY queue, state"
        ).fetchall()
        return {(r["queue"], JobState(r["state"])): r["n"] for r in rows}
