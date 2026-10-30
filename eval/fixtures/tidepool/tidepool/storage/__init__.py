"""Storage backends. ``open_storage`` picks one from the ``database`` setting."""

from __future__ import annotations

from tidepool.storage.base import Storage
from tidepool.storage.memory import MemoryStorage
from tidepool.storage.sqlite import SQLiteStorage


def open_storage(database: str) -> Storage:
    if database == ":memory:":
        return MemoryStorage()
    return SQLiteStorage(database)


__all__ = ["MemoryStorage", "SQLiteStorage", "Storage", "open_storage"]
