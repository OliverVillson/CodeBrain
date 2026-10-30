"""In-memory ``MemoryBackend`` for tests and for trying the server without Postgres.

It keeps the contract of the real backend (filters, token budgets, retrieval logging,
usage credit, duplicate merge, produced_by links) with deliberately simple parts:
ranking is identifier-aware term overlap, and ingest makes one module unit per file.
"""

from __future__ import annotations

import hashlib
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from codebrain.config import Settings, get_settings
from codebrain.models import (
    Edge,
    EdgeType,
    Episode,
    Namespace,
    Resolution,
    Retrieval,
    Unit,
    UnitKind,
    code_unit_id,
)
from codebrain.server.backend import UnknownJobError, UnknownUnitError
from codebrain.server.schemas import (
    ExplainResult,
    IngestJob,
    IngestRequest,
    JobState,
    LinkedUnit,
    RecalledUnit,
    RecallFilters,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
    StoreStatus,
)

LANGUAGES = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}

_WORD = re.compile(r"[A-Za-z][a-z]*|[A-Z]+(?![a-z])|\d+")


def terms(text: str) -> set[str]:
    """Lower-cased words with identifiers split: ``getUserById`` -> get, user, by, id."""
    return {w.lower() for w in _WORD.findall(text) if len(w) > 1}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class InMemoryBackend:
    name = "memory"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.units: dict[str, Unit] = {}
        self.edges: set[Edge] = set()
        self.episodes: dict[str, Episode] = {}
        self.retrievals: list[Retrieval] = []
        self.jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def add_unit(self, unit: Unit) -> None:
        with self._lock:
            self.units[unit.id] = unit

    def add_edge(self, edge: Edge) -> None:
        with self._lock:
            self.edges.add(edge)

    # recall

    def recall(self, request: RecallRequest) -> RecallResult:
        q = terms(request.query)
        with self._lock:
            scored: list[tuple[float, dict[str, float], Unit]] = []
            for u in self.units.values():
                if not _matches(u, request.filters):
                    continue
                features = _score(q, u)
                if features["lexical"] == 0:
                    continue
                score = features.pop("score")
                scored.append((score, features, u))
            scored.sort(key=lambda t: (-t[0], t[2].id))

            packed: list[RecalledUnit] = []
            used = 0
            for score, features, u in scored:
                if len(packed) == request.k:
                    break
                remaining = None if request.token_budget is None else request.token_budget - used
                view = _fit(u, remaining)
                if view is None:
                    continue
                resolution, content, tokens = view
                used += tokens
                packed.append(
                    RecalledUnit(
                        unit_id=u.id,
                        rank=len(packed) + 1,
                        score=round(score, 6),
                        resolution=resolution,
                        kind=u.kind,
                        content=content,
                        tokens=tokens,
                        path=u.path,
                        qualified_name=u.qualified_name,
                        span=u.span,
                        layer=u.layer,
                        summary=u.summary,
                        validity=u.validity,
                        features=features,
                    )
                )

            for r in packed:
                self.retrievals.append(
                    Retrieval(
                        query=request.query,
                        unit_id=r.unit_id,
                        session_id=request.session_id,
                        rank=r.rank,
                        score=r.score,
                        resolution=r.resolution,
                        features=r.features,
                    )
                )
                u = self.units[r.unit_id]
                self.units[u.id] = u.model_copy(update={"retrieved_count": u.retrieved_count + 1})

        return RecallResult(
            query=request.query, units=packed, tokens_used=used, token_budget=request.token_budget
        )

    # remember

    def remember(self, request: RememberRequest) -> RememberResult:
        kind = UnitKind(request.kind.value)
        namespace = Namespace.SESSION if request.session_only else Namespace.PROJECT
        text = f"{request.summary}\n\nTask: {request.prompt}"
        if request.pitfalls:
            text += "\n\nPitfalls:\n" + "\n".join(f"- {p}" for p in request.pitfalls)
        new_terms = terms(request.summary)

        with self._lock:
            referenced = dict.fromkeys(request.used_unit_ids + request.touched_unit_ids)
            unknown = [i for i in referenced if i not in self.units]

            duplicate = next(
                (
                    u
                    for u in self.units.values()
                    if u.kind is kind
                    and u.repo == request.repo
                    and _jaccard(new_terms, terms(u.summary or ""))
                    >= self.settings.write_back.dup_threshold
                ),
                None,
            )
            if duplicate is not None:
                unit_id = duplicate.id
                self.units[unit_id] = duplicate.model_copy(
                    update={"version": duplicate.version + 1, "updated_at": datetime.now(UTC)}
                )
            else:
                unit = Unit(
                    kind=kind,
                    namespace=namespace,
                    repo=request.repo,
                    session_id=request.session_id,
                    text=text,
                    summary=request.summary,
                    tokens=estimate_tokens(text),
                    capabilities=request.capabilities,
                    labels={"tests_passed": request.tests_passed},
                )
                unit_id = unit.id
                self.units[unit_id] = unit
                self.episodes[unit_id] = Episode(
                    unit_id=unit_id,
                    session_id=request.session_id,
                    prompt=request.prompt,
                    plan=request.plan,
                    diff=request.diff,
                    test_results=request.test_results,
                    tests_passed=request.tests_passed,
                    pitfalls=request.pitfalls,
                    started_at=request.started_at,
                )

            credited = [i for i in dict.fromkeys(request.used_unit_ids) if i in self.units]
            for i in credited:
                u = self.units[i]
                self.units[i] = u.model_copy(update={"used_count": u.used_count + 1})
            linked = [i for i in dict.fromkeys(request.touched_unit_ids) if i in self.units]
            for i in linked:
                self.edges.add(Edge(src_id=i, dst_id=unit_id, type=EdgeType.PRODUCED_BY))

        return RememberResult(
            unit_id=unit_id,
            merged_into=unit_id if duplicate is not None else None,
            credited=credited,
            linked=linked,
            unknown_unit_ids=unknown,
        )

    # explain

    def explain(self, unit_id: str, query: str | None = None) -> ExplainResult:
        with self._lock:
            unit = self.units.get(unit_id)
            if unit is None:
                raise UnknownUnitError(unit_id)
            outgoing = [
                self._linked(e, e.dst_id)
                for e in sorted(self.edges, key=_edge_key)
                if e.src_id == unit_id
            ]
            incoming = [
                self._linked(e, e.src_id)
                for e in sorted(self.edges, key=_edge_key)
                if e.dst_id == unit_id
            ]
            return ExplainResult(
                unit=unit,
                quality=unit.quality,
                outgoing=outgoing,
                incoming=incoming,
                episode=self.episodes.get(unit_id),
                query=query,
                trace=_score(terms(query), unit) if query else None,
            )

    def _linked(self, edge: Edge, other_id: str) -> LinkedUnit:
        other = self.units.get(other_id)
        return LinkedUnit(
            unit_id=other_id,
            type=edge.type,
            weight=edge.effective_weight,
            kind=other.kind if other else None,
            path=other.path if other else None,
            qualified_name=other.qualified_name if other else None,
            summary=other.summary if other else None,
        )

    # ingest and status

    def ingest(self, request: IngestRequest) -> IngestJob:
        job = IngestJob(
            job_id="job_" + uuid.uuid4().hex[:12],
            repo=request.repo,
            state=JobState.RUNNING,
            started_at=datetime.now(UTC),
        )
        with self._lock:
            self.jobs[job.job_id] = job
        try:
            job = self._ingest(request, job)
            job.state = JobState.DONE
        except OSError as e:
            job.state = JobState.FAILED
            job.errors.append(str(e))
        job.finished_at = datetime.now(UTC)
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def _ingest(self, request: IngestRequest, job: IngestJob) -> IngestJob:
        root = Path(request.path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {request.path}")
        scopes = [root / p for p in request.paths] if request.paths else [root]
        files: list[Path] = []
        for scope in scopes:
            candidates = [scope] if scope.is_file() else sorted(scope.rglob("*"))
            files.extend(
                f
                for f in candidates
                if f.is_file()
                and f.suffix in LANGUAGES
                and not SKIP_DIRS.intersection(f.relative_to(root).parts)
            )

        seen: set[str] = set()
        for f in files:
            job.files_seen += 1
            rel = f.relative_to(root).as_posix()
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                job.errors.append(f"{rel}: {e}")
                continue
            unit_id = code_unit_id(request.repo, rel, rel)
            seen.add(unit_id)
            digest = hashlib.sha256(text.encode()).hexdigest()
            with self._lock:
                old = self.units.get(unit_id)
                if old is not None and old.ast_hash == digest and not request.force:
                    job.units_unchanged += 1
                    continue
                first_line = next((line for line in text.splitlines() if line.strip()), rel)
                self.units[unit_id] = Unit(
                    id=unit_id,
                    kind=UnitKind.MODULE,
                    repo=request.repo,
                    path=rel,
                    qualified_name=rel,
                    language=LANGUAGES[f.suffix],
                    text=text,
                    summary=first_line.strip()[:200],
                    signature=rel,
                    lexical_terms=" ".join(sorted(terms(rel))),
                    ast_hash=digest,
                    tokens=estimate_tokens(text),
                    retrieved_count=old.retrieved_count if old else 0,
                    used_count=old.used_count if old else 0,
                    version=old.version + 1 if old else 1,
                )
                job.units_written += 1

        if not request.paths:
            with self._lock:
                gone = [
                    u.id
                    for u in self.units.values()
                    if u.repo == request.repo and u.kind is UnitKind.MODULE and u.id not in seen
                ]
                for i in gone:
                    del self.units[i]
                    self.edges = {e for e in self.edges if i not in (e.src_id, e.dst_id)}
                job.units_removed = len(gone)
        return job

    def status(self, job_id: str | None = None) -> StoreStatus:
        with self._lock:
            if job_id is not None and job_id not in self.jobs:
                raise UnknownJobError(job_id)
            by_kind: dict[str, int] = {}
            for u in self.units.values():
                by_kind[u.kind.value] = by_kind.get(u.kind.value, 0) + 1
            jobs = [self.jobs[job_id]] if job_id else list(self.jobs.values())
            return StoreStatus(
                backend=self.name,
                units_total=len(self.units),
                units_by_kind=dict(sorted(by_kind.items())),
                repos=sorted({u.repo for u in self.units.values() if u.repo}),
                episodes=len(self.episodes),
                jobs=jobs,
            )


def _unit_terms(u: Unit) -> set[str]:
    parts = [u.text, u.summary, u.signature, u.qualified_name, u.path, u.lexical_terms]
    return terms(" ".join(p for p in parts if p)) | {c.lower() for c in u.capabilities}


def _score(q: set[str], u: Unit) -> dict[str, float]:
    lexical = len(q & _unit_terms(u)) / len(q) if q else 0.0
    score = (0.7 * lexical + 0.3 * u.quality) * u.validity
    return {"lexical": lexical, "quality": u.quality, "validity": u.validity, "score": score}


def _matches(u: Unit, f: RecallFilters) -> bool:
    return (
        (f.kinds is None or u.kind in f.kinds)
        and (f.layers is None or u.layer in f.layers)
        and (f.namespaces is None or u.namespace in f.namespaces)
        and (f.repo is None or u.repo == f.repo)
        and (f.path_prefix is None or (u.path or "").startswith(f.path_prefix))
        and (f.capabilities is None or bool(set(f.capabilities) & set(u.capabilities)))
        and u.validity >= f.min_validity
    )


def _fit(u: Unit, remaining: int | None) -> tuple[Resolution, str, int] | None:
    """The richest view of ``u`` that fits in ``remaining`` tokens (sketch §7, stage 4)."""
    views: list[tuple[Resolution, str, int]] = [
        (Resolution.FULL, u.text, u.tokens or estimate_tokens(u.text))
    ]
    if u.summary:
        views.append((Resolution.SUMMARY, u.summary, estimate_tokens(u.summary)))
    sig = u.signature or u.qualified_name
    if sig:
        views.append((Resolution.SIGNATURE, sig, estimate_tokens(sig)))
    for view in views:
        if remaining is None or view[2] <= remaining:
            return view
    return None


def _edge_key(e: Edge) -> tuple[str, str, str]:
    return (e.type.value, e.src_id, e.dst_id)
