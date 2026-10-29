from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from codebrain.store import connect, migrate

DATABASE_URL = os.environ.get("CODEBRAIN_DATABASE_URL")


@pytest.fixture(scope="session")
def database_url() -> str:
    if not DATABASE_URL:
        pytest.skip("CODEBRAIN_DATABASE_URL not set")
    # Start every test run from an empty schema.
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    migrate(DATABASE_URL)
    return DATABASE_URL


@pytest.fixture
def conn(database_url: str) -> Iterator[psycopg.Connection]:
    """A connection whose work is rolled back after the test."""
    c = connect(database_url)
    try:
        yield c
    finally:
        c.rollback()
        c.close()
