"""Ingest: repo -> parse AST -> chunk -> summarize -> embed -> write (sketch §3-§5).

Entry point: ``Ingestor(store, repo).ingest(root, paths=None)``; see ``pipeline``.
Layer and capability classification (§5) is not part of ingest yet.
"""

from codebrain.ingest.pipeline import Ingestor, IngestReport
from codebrain.ingest.store import IngestStore, MemoryStore, PostgresStore

__all__ = ["IngestReport", "IngestStore", "Ingestor", "MemoryStore", "PostgresStore"]
