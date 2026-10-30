"""What write-back and paging need from the outside world.

The embedding and summary models are undecided (sketch §10), so the memory code takes
them as small protocols. ``MemoryStore`` is the slice of the store it reads and writes;
``memory.pg.PgMemoryStore`` implements it on Postgres, and tests use an in-memory fake.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from codebrain.models import Edge, Episode, Retrieval, Unit, UnitKind, UnitVector

if TYPE_CHECKING:
    from codebrain.memory.diff import FileChange
    from codebrain.memory.records import DistilledTask, TaskRecord, Turn


class Embedder(Protocol):
    """E_text from sketch §4. Vectors are normalized by the caller."""

    @property
    def model(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Distiller(Protocol):
    """Turns raw records into the searchable text of experience units (an LLM in practice)."""

    def distill_task(self, task: TaskRecord) -> DistilledTask: ...

    def distill_turns(self, turns: Sequence[Turn]) -> str: ...


class MemoryStore(Protocol):
    def get_units(self, unit_ids: Iterable[str]) -> dict[str, Unit]: ...

    def upsert_unit(self, unit: Unit) -> None: ...

    def upsert_vector(self, vec: UnitVector) -> None: ...

    def add_edges(self, edges: Iterable[Edge]) -> None: ...

    def insert_episode(self, episode: Episode) -> None: ...

    def nearest(
        self, kind: UnitKind, model: str, embedding: Sequence[float], repo_name: str | None
    ) -> tuple[Unit, float] | None:
        """The closest stored unit of ``kind`` by summary-vector cosine, for the duplicate check."""
        ...

    def units_changed_by(self, repo: str, changes: Sequence[FileChange]) -> list[Unit]:
        """The innermost code units whose spans overlap the changed lines."""
        ...

    def open_retrievals(self, session_id: str, since: datetime | None) -> list[Retrieval]: ...

    def mark_retrievals(self, used: dict[int, bool]) -> None: ...

    def add_usage(self, unit_id: str, *, retrieved: int, used: int) -> None: ...

    def set_validity(self, unit_ids: Iterable[str], validity: float) -> None: ...


TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    """Rough count (about four characters a token) for harnesses that don't pass their own."""
    return math.ceil(len(text) / 4)


def normalize(vec: Sequence[float]) -> list[float]:
    """‖v‖ = 1, so cosine is a dot product (sketch §4)."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else list(vec)
