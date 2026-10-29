"""Database connections. Every connection has the pgvector types registered."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress

import psycopg
from pgvector.psycopg import register_vector

from codebrain.config import get_settings


def connect(url: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(url or get_settings().database_url, autocommit=autocommit)
    # The vector extension is created by the first migration; callers that run
    # migrations on a fresh database connect before it exists.
    with suppress(psycopg.ProgrammingError):
        register_vector(conn)
    return conn


@contextmanager
def connection(url: str | None = None) -> Iterator[psycopg.Connection]:
    """Connection that commits on success and rolls back on error."""
    with connect(url) as conn:
        yield conn
