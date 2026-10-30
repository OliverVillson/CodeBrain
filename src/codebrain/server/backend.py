"""The interface the MCP server calls. Retrieval, write-back and ingest implement it.

The server only knows this protocol. ``memory.InMemoryBackend`` is a small fake for
tests and demos; the Postgres-backed implementation wires ``codebrain.retrieval``,
``codebrain.ingest`` and write-back behind the same methods.

Methods are synchronous because the store layer is. The server runs them in a
worker thread so a slow recall does not block the stdio loop.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from codebrain.server.schemas import (
    ExplainResult,
    IngestJob,
    IngestRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
    StoreStatus,
)


class UnknownUnitError(LookupError):
    """Raised by ``explain`` when no unit has the given id."""


class UnknownJobError(LookupError):
    """Raised by ``status`` when no ingest job has the given id."""


@runtime_checkable
class MemoryBackend(Protocol):
    name: str

    def recall(self, request: RecallRequest) -> RecallResult:
        """Rank, pack and log units for a query (sketch §7).

        Implementations log every returned unit as a ``Retrieval`` so write-back can
        credit it later.
        """
        ...

    def remember(self, request: RememberRequest) -> RememberResult:
        """Store a finished task as an experience unit and credit what helped (sketch §8)."""
        ...

    def explain(self, unit_id: str, query: str | None = None) -> ExplainResult:
        """Return a unit with its graph neighbours and evidence. Raises ``UnknownUnitError``.

        With ``query``, also fill ``trace`` with how the unit scores for that query.
        """
        ...

    def ingest(self, request: IngestRequest) -> IngestJob:
        """Start indexing a checkout. May return before the job finishes."""
        ...

    def status(self, job_id: str | None = None) -> StoreStatus:
        """Store counts and ingest jobs; only ``job_id`` when given. Raises ``UnknownJobError``."""
        ...
