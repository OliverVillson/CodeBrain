"""The ingest loop: files -> chunks -> summaries -> vectors -> units and edges (sketch §3).

``Ingestor.ingest(root)`` scans a checkout; ``ingest(root, paths=[...])`` re-ingests
only the named files or directories (a post-commit hook or file watcher passes the
changed paths). Either way, work is skipped at three levels:

* a file whose bytes hash the same as at the last ingest, under the same summarizer
  and embedding models, is not parsed again;
* a unit whose ``ast_hash`` is unchanged keeps its summary;
* a vector whose input text hashes the same is not re-embedded, and an input seen
  on any other unit (moved or copied code) reuses that unit's vector.

Units that disappear, with their file or on their own, are tombstoned (see
``codebrain.ingest.store``), and episodes or decisions linked to changed or removed
code drop to validity 0.3.

Calls into a re-ingested file from files that were not re-ingested keep pointing at
the ids they resolved to before. When a target is renamed, the old edge goes with
the tombstone and the caller links to the new name on its own next re-ingest, or on
the next full scan.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import posixpath
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codebrain.config import ChunkingParams, Settings, get_settings
from codebrain.ingest.chunker import Chunk, Chunker, ParsedFile, hash_text
from codebrain.ingest.embed import Embedder, make_embedder
from codebrain.ingest.graph import GraphBuilder, SymbolTable
from codebrain.ingest.languages import language_for
from codebrain.ingest.lexical import lexical_terms
from codebrain.ingest.store import IngestStore, is_tombstoned
from codebrain.ingest.summarize import (
    HeuristicSummarizer,
    Summarizer,
    SummaryRequest,
    make_summarizer,
)
from codebrain.models import (
    Namespace,
    Span,
    Unit,
    UnitVector,
    VectorView,
    code_unit_id,
)

log = logging.getLogger(__name__)

INGEST_VERSION = 1
"""Bump when chunking changes, so every file is re-parsed on the next run."""

INGEST_LABELS = frozenset(
    {
        "file_hash",
        "ingest_version",
        "ingest_models",
        "has_errors",
        "is_test",
        "summarizer",
        "merged",
        "part",
        "parts",
        "split",
        "skeleton",
        "truncated",
        "tombstoned",
        "tombstoned_at",
    }
)
"""Label keys ingest owns. Other keys on a stored unit (from classifiers, agents) are kept."""


@dataclass
class IngestReport:
    repo: str
    files_seen: int = 0
    files_unchanged: int = 0
    files_parsed: int = 0
    files_removed: int = 0
    units_written: int = 0
    units_unchanged: int = 0
    """Written again (path or lines may move) but with the same ``ast_hash``."""
    units_removed: int = 0
    summaries_generated: int = 0
    summaries_reused: int = 0
    vectors_embedded: int = 0
    vectors_reused: int = 0
    edges_written: int = 0
    linked_marked_stale: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class _FileWork:
    pf: ParsedFile
    old: dict[str, Unit]
    file_hash: str
    force: bool = False
    units: dict[str, Unit] = field(default_factory=dict)
    summaries: dict[str, str] = field(default_factory=dict)
    summarizer_of: dict[str, str] = field(default_factory=dict)


class Ingestor:
    def __init__(
        self,
        store: IngestStore,
        repo: str,
        *,
        settings: Settings | None = None,
        summarizer: Summarizer | None = None,
        embedders: Mapping[VectorView, Embedder] | None = None,
        chunker: Chunker | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        s = settings or get_settings()
        self.store = store
        self.repo = repo
        self.params: ChunkingParams = s.chunking
        self.batch_size = s.embedding.batch_size
        self.chunker = chunker or Chunker(s.chunking.tau_min, s.chunking.tau_max)
        self.summarizer = summarizer or make_summarizer(
            s.summary.model,
            **(
                {}
                if s.summary.model == "heuristic"
                else {
                    "effort": s.summary.effort,
                    "max_tokens": s.summary.max_tokens,
                    "concurrency": s.summary.concurrency,
                }
            ),
        )
        if embedders is None:
            embedders = {
                VectorView.SUMMARY: make_embedder(
                    s.embedding.text_model, s.embedding.dim, s.embedding.batch_size
                ),
                VectorView.CODE: make_embedder(
                    s.embedding.code_model, s.embedding.dim, s.embedding.batch_size
                ),
            }
        self.embedders = dict(embedders)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._fallback = HeuristicSummarizer()

    @property
    def models(self) -> dict[str, str]:
        """The summarizer and embedding models a file was last ingested with."""
        out = {"summarizer": self.summarizer.name}
        out.update({str(view): e.model for view, e in sorted(self.embedders.items())})
        return out

    # Entry point

    def ingest(
        self, root: str | Path, paths: Sequence[str] | None = None, *, force: bool = False
    ) -> IngestReport:
        """Ingest the checkout at ``root``, or only ``paths`` (files or dirs relative to it)."""
        root = Path(root).resolve()
        report = IngestReport(repo=self.repo)
        table = SymbolTable()
        for s in self.store.symbols(self.repo):
            table.add(s.symbol, s.aliases)
        stored_paths = table.paths

        on_disk = discover(root, self.params)
        if paths is None:
            candidates = on_disk
            removed = sorted(stored_paths - set(on_disk))
        else:
            scopes = [_scope(root, p) for p in paths]
            candidates = [p for p in on_disk if any(_within(p, s) for s in scopes)]
            disk = set(on_disk)
            removed = sorted(
                p for p in stored_paths if p not in disk and any(_within(p, s) for s in scopes)
            )

        work: list[_FileWork] = []
        for path in candidates:
            report.files_seen += 1
            try:
                w = self._prepare(root, path, force, report)
            except Exception as e:  # keep going: one unparsable file must not stop the run
                log.exception("ingest failed for %s", path)
                report.errors.append(f"{path}: {e}")
                continue
            if w is not None:
                work.append(w)

        for path in removed:
            table.remove_path(path)
        for w in work:
            table.remove_path(w.pf.path)
            table.add_file(self.repo, w.pf)

        self._summarize(work, report)
        self._build_units(work)
        vectors = self._embed(work, report)

        now = self.clock()
        changed: list[str] = []
        dead: list[str] = []
        for w in work:
            for u in w.units.values():  # chunk order: parents first, for the parent_id FK
                self.store.upsert_unit(u)
                report.units_written += 1
                old = w.old.get(u.id)
                if old is not None and not is_tombstoned(old):
                    if old.ast_hash == u.ast_hash:
                        report.units_unchanged += 1
                    else:
                        changed.append(u.id)
            dead.extend(
                uid for uid, old in w.old.items() if uid not in w.units and not is_tombstoned(old)
            )
        for v in vectors:
            self.store.upsert_vector(v)

        graph = GraphBuilder(self.repo, table)
        for w in work:
            edges = graph.edges_for(w.pf)
            self.store.replace_edges(list(w.units), edges)
            report.edges_written += len(edges)

        for path in removed:
            live = [u.id for u in self.store.units_at(self.repo, path) if not is_tombstoned(u)]
            if live:
                report.files_removed += 1
                dead.extend(live)
        self.store.tombstone(dead, now)
        report.units_removed = len(dead)
        report.linked_marked_stale = self.store.mark_linked_stale(changed + dead)
        return report

    # Steps

    def _prepare(
        self, root: Path, path: str, force: bool, report: IngestReport
    ) -> _FileWork | None:
        data = (root / path).read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()[:32]
        old = {u.id: u for u in self.store.units_at(self.repo, path)}
        module = old.get(code_unit_id(self.repo, path, path))
        if (
            not force
            and module is not None
            and not is_tombstoned(module)
            and module.labels.get("file_hash") == file_hash
            and module.labels.get("ingest_version") == INGEST_VERSION
            # A new summarizer or embedder must still reach unchanged files; reparsing is
            # cheap, and the per-unit summary and vector checks skip what is still current.
            and module.labels.get("ingest_models") == self.models
        ):
            report.files_unchanged += 1
            return None
        spec = language_for(path)
        if spec is not None:
            pf = self.chunker.chunk_code(path, data, spec)
        else:
            pf = self.chunker.chunk_prompt(path, data.decode("utf-8", "replace"))
        report.files_parsed += 1
        return _FileWork(pf, old, file_hash, force)

    def _summarize(self, work: list[_FileWork], report: IngestReport) -> None:
        """Bottom-up: deepest chunks first, so parents see their children's summaries."""
        by_depth: dict[int, list[tuple[_FileWork, Chunk]]] = defaultdict(list)
        for w in work:
            for c in w.pf.chunks:
                by_depth[c.depth].append((w, c))
        for depth in sorted(by_depth, reverse=True):
            todo: list[tuple[_FileWork, Chunk, SummaryRequest]] = []
            for w, c in by_depth[depth]:
                uid = code_unit_id(self.repo, w.pf.path, c.qualified_name)
                old = w.old.get(uid)
                if (
                    not w.force
                    and old is not None
                    and old.summary
                    and old.ast_hash == c.ast_hash
                    and old.labels.get("summarizer") == self.summarizer.name
                ):
                    w.summaries[c.qualified_name] = old.summary
                    w.summarizer_of[c.qualified_name] = self.summarizer.name
                    report.summaries_reused += 1
                    continue
                children = tuple(
                    (k.qualified_name, w.summaries.get(k.qualified_name, ""))
                    for k in w.pf.chunks
                    if k.parent == c.qualified_name
                )
                todo.append((w, c, self._request(w.pf, c, children)))
            if not todo:
                continue
            results = self.summarizer.summarize([r for _, _, r in todo])
            for (w, c, req), text in zip(todo, results, strict=True):
                name = self.summarizer.name
                if not text:
                    text, name = self._fallback.one(req), self._fallback.name
                w.summaries[c.qualified_name] = text
                w.summarizer_of[c.qualified_name] = name
                report.summaries_generated += 1

    def _request(
        self, pf: ParsedFile, c: Chunk, children: tuple[tuple[str, str], ...]
    ) -> SummaryRequest:
        return SummaryRequest(
            kind=c.kind,
            qualified_name=c.qualified_name,
            path=pf.path,
            language=pf.language,
            text=c.text,
            signature=c.signature,
            children=children,
        )

    def _build_units(self, work: list[_FileWork]) -> None:
        now = self.clock()
        for w in work:
            pf = w.pf
            for i, c in enumerate(pf.chunks):
                uid = code_unit_id(self.repo, pf.path, c.qualified_name)
                old = w.old.get(uid)
                labels: dict[str, Any] = {
                    k: v for k, v in (old.labels if old else {}).items() if k not in INGEST_LABELS
                }
                labels.update(c.labels)
                labels["summarizer"] = w.summarizer_of.get(c.qualified_name)
                if pf.is_test:
                    labels["is_test"] = True
                if i == 0:
                    labels.update(
                        file_hash=w.file_hash,
                        ingest_version=INGEST_VERSION,
                        ingest_models=self.models,
                    )
                    if pf.has_errors:
                        labels["has_errors"] = True
                version = 1
                if old is not None:
                    version = old.version + (1 if old.ast_hash != c.ast_hash else 0)
                w.units[uid] = Unit(
                    id=uid,
                    kind=c.kind,
                    namespace=Namespace.CODE,
                    repo=self.repo,
                    path=pf.path,
                    qualified_name=c.qualified_name,
                    language=pf.language,
                    span=Span(start_line=c.start_line, end_line=max(c.start_line, c.end_line)),
                    parent_id=code_unit_id(self.repo, pf.path, c.parent) if c.parent else None,
                    text=c.text,
                    summary=w.summaries.get(c.qualified_name),
                    signature=c.signature,
                    lexical_terms=lexical_terms(c.text, names=[c.qualified_name, pf.path]),
                    ast_hash=c.ast_hash,
                    tokens=c.tokens,
                    layer=old.layer if old else None,
                    layer_confidence=old.layer_confidence if old else None,
                    capabilities=old.capabilities if old else [],
                    labels=labels,
                    version=version,
                    created_at=old.created_at if old else now,
                    updated_at=now,
                )

    def _embed(self, work: list[_FileWork], report: IngestReport) -> list[UnitVector]:
        out: list[UnitVector] = []
        for view, embedder in self.embedders.items():
            queue: list[tuple[str, str, str]] = []  # unit id, input, content hash
            # One query for every stored vector hash, not one per unit.
            known = self.store.vector_hashes(
                [uid for w in work if not w.force for uid in w.units if uid in w.old],
                view,
                embedder.model,
            )
            for w in work:
                for uid, u in w.units.items():
                    text = embedding_input(u, view)
                    ch = hash_text(text)
                    if w.force:
                        queue.append((uid, text, ch))
                        continue
                    if uid in w.old and known.get(uid) == ch:
                        report.vectors_reused += 1
                        continue
                    found = self.store.find_vector(view, embedder.model, ch)
                    if found is not None:
                        out.append(_vec(uid, view, embedder.model, found, ch))
                        report.vectors_reused += 1
                    else:
                        queue.append((uid, text, ch))
            for batch in _batches(queue, self.batch_size):
                embs = embedder.embed([t for _, t, _ in batch])
                for (uid, _, ch), emb in zip(batch, embs, strict=True):
                    out.append(_vec(uid, view, embedder.model, emb, ch))
                report.vectors_embedded += len(batch)
        return out


def embedding_input(u: Unit, view: VectorView) -> str:
    """s_u embeds the summary (English); c_u embeds the source (sketch §4)."""
    if view is VectorView.SUMMARY:
        return u.summary or u.text
    return u.text


def _vec(uid: str, view: VectorView, model: str, emb: Sequence[float], ch: str) -> UnitVector:
    return UnitVector(unit_id=uid, view=view, model=model, embedding=list(emb), content_hash=ch)


def _batches(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), max(1, n)):
        yield items[i : i + n]


# File discovery


def discover(root: Path, params: ChunkingParams) -> list[str]:
    """Repo-relative POSIX paths of files to ingest: code in a known language, or prompts."""
    files = _git_files(root)
    if files is None:
        files = [
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
        ]
    out = []
    for rel in sorted(set(files)):
        if _matches(rel, params.exclude_globs):
            continue
        if language_for(rel) is None and not _matches(rel, params.prompt_globs):
            continue
        full = root / rel
        try:
            if not full.is_file() or full.stat().st_size > params.max_file_bytes:
                continue
            with full.open("rb") as fh:
                if b"\x00" in fh.read(8192):
                    continue
        except OSError:
            continue
        out.append(rel)
    return out


def _git_files(root: Path) -> list[str] | None:
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode().split("\0") if p]


def _matches(path: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch("/" + path, g) for g in globs)


def _norm(p: str) -> str:
    return posixpath.normpath(Path(p).as_posix()).strip("/").removeprefix("./") or "."


def _scope(root: Path, p: str) -> str:
    """A ``paths`` entry as a repo-relative path. Absolute paths must lie under ``root``."""
    path = Path(p).expanduser()
    if path.is_absolute():
        try:
            return _norm(path.resolve().relative_to(root).as_posix())
        except ValueError:
            raise ValueError(f"{p} is outside the checkout {root}") from None
    scope = _norm(p)
    if scope == ".." or scope.startswith("../"):
        raise ValueError(f"{p} is outside the checkout {root}")
    return scope


def _within(path: str, scope: str) -> bool:
    return scope == "." or path == scope or path.startswith(scope + "/")
