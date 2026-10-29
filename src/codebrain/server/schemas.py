"""Tool inputs and outputs for the MCP server (sketch §10).

These are the wire shapes agents see. They are built from ``codebrain.models`` so a
backend can fill them straight from stored rows.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from codebrain.models import (
    EdgeType,
    Episode,
    Layer,
    Namespace,
    Resolution,
    Span,
    Unit,
    UnitKind,
)

RECALL_MAX_K = 50


class RecallFilters(BaseModel):
    """Hard filters applied before ranking. Every set field must match."""

    kinds: list[UnitKind] | None = Field(default=None, description="Only these unit kinds.")
    layers: list[Layer] | None = Field(default=None, description="Only these layers.")
    namespaces: list[Namespace] | None = Field(default=None, description="Only these namespaces.")
    repo: str | None = Field(default=None, description="Only units from this repo.")
    path_prefix: str | None = Field(
        default=None, description="Only units whose path starts with this."
    )
    capabilities: list[str] | None = Field(
        default=None, description="Only units tagged with at least one of these capabilities."
    )
    min_validity: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Drop units below this v(u). 0.3 hides deprecated units; 1.0 keeps only "
        "units whose code is unchanged.",
    )


class RecallRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=RECALL_MAX_K)
    token_budget: int | None = Field(default=None, ge=1)
    filters: RecallFilters = Field(default_factory=RecallFilters)
    session_id: str | None = None


class RecalledUnit(BaseModel):
    """One packed unit. ``content`` is the unit shown at ``resolution``."""

    unit_id: str
    rank: int = Field(ge=1)
    score: float
    resolution: Resolution
    kind: UnitKind
    content: str
    tokens: int = Field(ge=0)
    path: str | None = None
    qualified_name: str | None = None
    span: Span | None = None
    layer: Layer | None = None
    summary: str | None = None
    validity: float
    features: dict[str, float] = Field(default_factory=dict)


class RecallResult(BaseModel):
    query: str
    units: list[RecalledUnit]
    tokens_used: int = Field(ge=0)
    token_budget: int | None = None


class RememberKind(StrEnum):
    EPISODE = UnitKind.EPISODE.value
    DECISION = UnitKind.DECISION.value


class RememberRequest(BaseModel):
    """A finished task, written back as an experience unit (sketch §8)."""

    summary: str = Field(
        min_length=1,
        description="The distilled lesson: what was built or decided, how, and why. This is "
        "the searchable text.",
    )
    prompt: str = Field(min_length=1, description="The task as the agent was given it.")
    kind: RememberKind = RememberKind.EPISODE
    plan: str | None = None
    diff: str | None = None
    tests_passed: bool | None = None
    test_results: dict[str, Any] = Field(default_factory=dict)
    pitfalls: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    repo: str | None = None
    session_id: str | None = None
    session_only: bool = Field(
        default=False,
        description="Store in the session namespace (short half-life) instead of project.",
    )
    used_unit_ids: list[str] = Field(
        default_factory=list,
        description="Recalled units that actually helped. Each gets usage credit.",
    )
    touched_unit_ids: list[str] = Field(
        default_factory=list,
        description="Code units the diff created or changed. Each is linked to this episode.",
    )
    started_at: datetime | None = None


class RememberResult(BaseModel):
    unit_id: str
    merged_into: str | None = Field(
        default=None,
        description="Set when a near-duplicate already existed and this was merged into it.",
    )
    credited: list[str] = Field(default_factory=list)
    linked: list[str] = Field(default_factory=list)
    unknown_unit_ids: list[str] = Field(default_factory=list)


class LinkedUnit(BaseModel):
    """A neighbour in the code graph, named so the agent can read it without a lookup."""

    unit_id: str
    type: EdgeType
    weight: float
    kind: UnitKind | None = None
    path: str | None = None
    qualified_name: str | None = None
    summary: str | None = None


class ExplainResult(BaseModel):
    unit: Unit
    quality: float = Field(description="Q(u) = (used + 1) / (retrieved + 2).")
    outgoing: list[LinkedUnit] = Field(default_factory=list)
    incoming: list[LinkedUnit] = Field(default_factory=list)
    episode: Episode | None = None


class IngestRequest(BaseModel):
    repo: str = Field(min_length=1, description="Repo name used in unit ids.")
    path: str = Field(min_length=1, description="Local checkout root to read files from.")
    paths: list[str] | None = Field(
        default=None, description="Limit to these files or directories, relative to path."
    )
    force: bool = Field(default=False, description="Re-embed units whose AST hash is unchanged.")


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class IngestJob(BaseModel):
    job_id: str
    repo: str
    state: JobState
    files_seen: int = 0
    units_written: int = 0
    units_unchanged: int = 0
    units_removed: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class StoreStatus(BaseModel):
    backend: str
    units_total: int
    units_by_kind: dict[str, int]
    repos: list[str]
    episodes: int
    jobs: list[IngestJob] = Field(default_factory=list)
