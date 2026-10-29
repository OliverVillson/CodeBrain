"""Settings and the tunable constants from the design sketch.

Every number here is a starting value from the sketch; the section is noted next
to each group. Override any of them with environment variables, for example
``CODEBRAIN_DATABASE_URL`` or ``CODEBRAIN_RETRIEVAL__W_SEMANTIC=0.4``.
"""

from __future__ import annotations

import math
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingParams(BaseModel):
    """Sketch §3."""

    tau_min: int = 64
    tau_max: int = 1200
    max_file_bytes: int = 1_000_000
    prompt_globs: list[str] = [
        "prompts/**",
        "**/prompts/**",
        "**/*.prompt",
        "**/*.prompt.md",
        "**/CLAUDE.md",
        "**/AGENTS.md",
    ]
    exclude_globs: list[str] = [
        "**/node_modules/**",
        "**/.venv/**",
        "**/venv/**",
        "**/vendor/**",
        "**/dist/**",
        "**/build/**",
        "**/target/**",
        "**/*.min.js",
        "**/*_pb2.py",
    ]


class SummaryParams(BaseModel):
    """Sketch §3: ``heuristic`` (no model), ``anthropic`` or ``anthropic:<model id>``."""

    model: str = "heuristic"
    effort: str = "low"
    max_tokens: int = 2048
    concurrency: int = 4


class EmbeddingParams(BaseModel):
    """Sketch §4. Models are undecided; these are placeholders until the eval set picks them."""

    text_model: str = "unset-text-model"
    code_model: str = "unset-code-model"
    dim: int = 1024
    prefilter_dim: int = 256
    batch_size: int = 64


class ClassificationParams(BaseModel):
    """Sketch §5."""

    temperature: float = 0.05
    accept_threshold: float = 0.5
    spread_alpha: float = 0.6
    spread_iterations: int = 10


class GraphParams(BaseModel):
    """Sketch §6."""

    ppr_beta: float = 0.5
    similar_to_threshold: float = 0.85


class RetrievalParams(BaseModel):
    """Sketch §7."""

    k_per_index: int = 50
    rrf_k: int = 60
    fused_keep: int = 100
    w_semantic: float = 0.35
    w_code: float = 0.15
    w_lexical: float = 0.15
    w_graph: float = 0.10
    w_quality: float = 0.15
    w_freshness: float = 0.10
    freshness_half_life_days: float = 30.0
    mmr_mu: float = 0.7
    cross_encoder_top: int = 30

    @property
    def freshness_lambda(self) -> float:
        """λ per day, so F(u) = exp(-λ Δt_days)."""
        return math.log(2) / self.freshness_half_life_days


class WriteBackParams(BaseModel):
    """Sketch §8."""

    dup_threshold: float = 0.92


class PagingParams(BaseModel):
    """Sketch §9."""

    high_water: float = 0.70
    low_water: float = 0.50
    session_weight: float = 1.2
    session_half_life_hours: float = 2.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CODEBRAIN_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    database_url: str = "postgresql://codebrain:codebrain@localhost:5432/codebrain"

    chunking: ChunkingParams = ChunkingParams()
    summary: SummaryParams = SummaryParams()
    embedding: EmbeddingParams = EmbeddingParams()
    classification: ClassificationParams = ClassificationParams()
    graph: GraphParams = GraphParams()
    retrieval: RetrievalParams = RetrievalParams()
    write_back: WriteBackParams = WriteBackParams()
    paging: PagingParams = PagingParams()


@lru_cache
def get_settings() -> Settings:
    return Settings()
