"""Core data models shared by every module.

These mirror the tables in ``store/migrations`` one to one. The design sketch
section each model comes from is noted on the class.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now() -> datetime:
    return datetime.now(UTC)


class UnitKind(StrEnum):
    """What a unit is (sketch §2). Code kinds come from the parser, the rest from agents."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    PROMPT = "prompt"
    EPISODE = "episode"
    DECISION = "decision"
    TURN = "turn"

    @property
    def is_code(self) -> bool:
        return self in _CODE_KINDS


_CODE_KINDS = {UnitKind.FUNCTION, UnitKind.METHOD, UnitKind.CLASS, UnitKind.MODULE, UnitKind.PROMPT}


class Namespace(StrEnum):
    """Where a unit lives. Session units use a shorter freshness half-life (sketch §9)."""

    CODE = "code"
    PROJECT = "project"
    SESSION = "session"


class Layer(StrEnum):
    """Closed set of architectural layers (sketch §5)."""

    UI = "ui"
    API = "api"
    DOMAIN = "domain"
    DATA = "data"
    INFRA = "infra"
    TEST = "test"
    TOOLING = "tooling"


class EdgeType(StrEnum):
    """Code graph edge types (sketch §6). ``default_weight`` is the starting PageRank weight."""

    CALLS = "calls"
    IMPORTS = "imports"
    CONTAINS = "contains"
    TESTS = "tests"
    PRODUCED_BY = "produced_by"
    SIMILAR_TO = "similar_to"

    @property
    def default_weight(self) -> float:
        return EDGE_WEIGHTS[self]


EDGE_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.CALLS: 1.0,
    EdgeType.IMPORTS: 0.6,
    EdgeType.CONTAINS: 0.5,
    EdgeType.TESTS: 0.8,
    EdgeType.PRODUCED_BY: 0.9,
    EdgeType.SIMILAR_TO: 0.4,
}


class VectorView(StrEnum):
    """Dense views of a unit (sketch §4). The lexical view is a tsvector on ``units``."""

    SUMMARY = "summary"
    CODE = "code"


# v(u) validity factors (sketch §7).
VALIDITY_CURRENT = 1.0
"""Code at HEAD is unchanged since the unit was written."""
VALIDITY_CHANGED = 0.3
"""Code changed or was removed; lessons may still apply."""
VALIDITY_DEPRECATED = 0.0
"""Superseded by a later decision."""


class Resolution(StrEnum):
    """How much of a unit is shown when packed into context (sketch §7, stage 4)."""

    SIGNATURE = "signature"
    SUMMARY = "summary"
    FULL = "full"

    @property
    def value_factor(self) -> float:
        return {"signature": 0.4, "summary": 0.7, "full": 1.0}[self.value]


class Span(BaseModel):
    """1-based inclusive line range in ``path``."""

    model_config = ConfigDict(frozen=True)

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


def code_unit_id(repo: str, path: str, qualified_name: str) -> str:
    """Stable id for a code unit: H(repo ‖ path ‖ qualified_name) (sketch §3)."""
    raw = f"{repo}\x00{path}\x00{qualified_name}".encode()
    return "u_" + hashlib.sha256(raw).hexdigest()[:32]


def new_unit_id() -> str:
    """Random id for experience units (episodes, decisions, turns)."""
    return "u_" + uuid.uuid4().hex


class Unit(BaseModel):
    """The single stored record (sketch §2). Code and experience share this schema."""

    id: str = Field(default_factory=new_unit_id)
    kind: UnitKind
    namespace: Namespace = Namespace.CODE
    repo: str | None = None
    session_id: str | None = None

    path: str | None = None
    qualified_name: str | None = None
    language: str | None = None
    span: Span | None = None
    parent_id: str | None = None

    text: str
    summary: str | None = None
    signature: str | None = None
    lexical_terms: str | None = Field(
        default=None, description="Split identifiers and keywords, fed to the lexical index."
    )

    ast_hash: str | None = Field(
        default=None, description="H(normalize(AST(u))), for change detection."
    )
    tokens: int = Field(default=0, ge=0)

    layer: Layer | None = None
    layer_confidence: float | None = Field(default=None, ge=0, le=1)
    capabilities: list[str] = Field(default_factory=list)
    labels: dict[str, Any] = Field(default_factory=dict)

    validity: float = Field(default=VALIDITY_CURRENT, ge=0, le=1)
    retrieved_count: int = Field(default=0, ge=0)
    used_count: int = Field(default=0, ge=0)
    version: int = Field(default=1, ge=1)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _check_namespace(self) -> Unit:
        if self.namespace is Namespace.SESSION and not self.session_id:
            raise ValueError("session units need a session_id")
        if self.kind.is_code and not self.path:
            raise ValueError(f"{self.kind} units need a path")
        return self

    @property
    def quality(self) -> float:
        """Q(u) = (used + 1) / (retrieved + 2), a Beta(1,1) prior (sketch §7)."""
        return (self.used_count + 1) / (self.retrieved_count + 2)


class UnitVector(BaseModel):
    """One embedding of one unit. The model id is stored so the index can be rebuilt."""

    unit_id: str
    view: VectorView
    model: str
    embedding: list[float]
    content_hash: str | None = Field(
        default=None, description="Hash of the embedded input, so unchanged units reuse vectors."
    )
    created_at: datetime = Field(default_factory=_now)

    @property
    def dim(self) -> int:
        return len(self.embedding)


class Edge(BaseModel):
    """A directed, typed edge in the code graph (sketch §6)."""

    model_config = ConfigDict(frozen=True)

    src_id: str
    dst_id: str
    type: EdgeType
    weight: float | None = None

    @property
    def effective_weight(self) -> float:
        return self.weight if self.weight is not None else self.type.default_weight


class Episode(BaseModel):
    """The raw record of a finished agent task (sketch §8).

    The distilled, searchable text lives on the matching ``Unit`` (kind ``episode``);
    this holds the evidence behind it.
    """

    unit_id: str
    session_id: str | None = None
    prompt: str
    plan: str | None = None
    diff: str | None = None
    test_results: dict[str, Any] = Field(default_factory=dict)
    tests_passed: bool | None = None
    pitfalls: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime = Field(default_factory=_now)


class Retrieval(BaseModel):
    """One unit returned by one recall, logged for usage credit and weight learning (sketch §8)."""

    id: int | None = None
    query: str
    unit_id: str
    session_id: str | None = None
    rank: int = Field(ge=1)
    score: float
    resolution: Resolution
    features: dict[str, float] = Field(
        default_factory=dict, description="Stage-2 feature values, the training input for weights."
    )
    used: bool | None = None
    created_at: datetime = Field(default_factory=_now)
