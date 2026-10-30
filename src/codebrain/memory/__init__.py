"""Write-back and context paging (sketch §8, §9).

- ``EpisodeRecorder`` turns a finished task into an episode unit, decision units and
  usage credit for the units the task reused.
- ``SessionTracker`` is the harness-facing API for a live context window: it pages the
  oldest turns out to the store as turn units instead of compacting them.
- ``PgMemoryStore`` is the Postgres backing for both.
"""

from codebrain.memory.credit import CreditReport, apply_credit
from codebrain.memory.diff import FileChange, parse_diff
from codebrain.memory.distill import PlainDistiller
from codebrain.memory.interfaces import Distiller, Embedder, MemoryStore, approx_tokens
from codebrain.memory.paging import (
    ContextMessage,
    PagedSpan,
    PageEvent,
    PinKind,
    Recalled,
    SessionState,
    SessionTracker,
)
from codebrain.memory.pg import PgMemoryStore
from codebrain.memory.records import (
    DecisionRecord,
    DistilledTask,
    Outcome,
    Role,
    TaskRecord,
    Turn,
)
from codebrain.memory.writeback import EpisodeRecorder, WriteBackResult

__all__ = [
    "ContextMessage",
    "CreditReport",
    "DecisionRecord",
    "DistilledTask",
    "Distiller",
    "Embedder",
    "EpisodeRecorder",
    "FileChange",
    "MemoryStore",
    "Outcome",
    "PageEvent",
    "PagedSpan",
    "PgMemoryStore",
    "PinKind",
    "PlainDistiller",
    "Recalled",
    "Role",
    "SessionState",
    "SessionTracker",
    "TaskRecord",
    "Turn",
    "WriteBackResult",
    "approx_tokens",
    "apply_credit",
    "parse_diff",
]
