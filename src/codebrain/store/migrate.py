"""Forward-only SQL migrations.

Migrations are the ``NNNN_name.sql`` files in ``store/migrations``, applied in
order, each in its own transaction. Applied versions are recorded in
``schema_migrations``. A new feature that needs schema adds the next number;
never edit a migration that has merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources

import psycopg

from codebrain.store.db import connect

_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
_LOCK_KEY = 0x0C0DEB7A1  # advisory lock so concurrent runners don't race


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


def discover() -> list[Migration]:
    found: list[Migration] = []
    for entry in resources.files("codebrain.store.migrations").iterdir():
        match = _NAME.match(entry.name)
        if match:
            found.append(Migration(int(match.group(1)), entry.name, entry.read_text()))
    found.sort(key=lambda m: m.version)
    versions = [m.version for m in found]
    if len(versions) != len(set(versions)):
        raise RuntimeError(f"duplicate migration numbers: {versions}")
    return found


def applied(conn: psycopg.Connection) -> set[int]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {r[0] for r in rows}


def migrate(url: str | None = None) -> list[str]:
    """Apply pending migrations. Returns the names applied."""
    done: list[str] = []
    with connect(url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version    integer PRIMARY KEY,
                    name       text NOT NULL,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            have = applied(conn)
            for m in discover():
                if m.version in have:
                    continue
                with conn.transaction():
                    conn.execute(m.sql.encode())
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                        (m.version, m.name),
                    )
                done.append(m.name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    return done
