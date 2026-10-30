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
    ppr_hops: int = 2
    """How far from the seed hits edges are loaded for the walk."""
    ppr_iterations: int = 50
    ppr_tolerance: float = 1e-6
    expand_keep: int = 20
    """Graph neighbours (not found by any index) added to the candidates, by PageRank."""
    expand_load_factor: int = 4
    """Neighbours are loaded in PageRank order, ``expand_load_factor * expand_keep`` at a
    time, until ``expand_keep`` pass the filters; the 2-hop neighbourhood can be most of
    the repo, so loading all of it (with vectors) is the main cost of a recall."""
    ppr_seeds: int = 10
    """Only the best fused hits seed the walk; seeding all ``fused_keep`` spread the mass
    over most of the repo."""
    hub_degree: int = 40
    """Units with more edges than this (a big module's ``contains`` fan-out) are not
    walked through on later hops, and their ``contains`` edges are down-weighted to
    ``hub_degree / degree`` of their weight, so a file is not a highway to all its
    siblings."""
    max_nodes: int = 2000
    """Stop loading more hops once the neighbourhood has this many units."""


class RetrievalParams(BaseModel):
    """Sketch §7."""

    k_per_index: int = 50
    rrf_k: int = 60
    fused_keep: int = 100
    w_semantic: float = 0.35
    w_code: float = 0.15
    w_lexical: float = 0.15
    w_graph: float = 0.10
    w_symbol: float = 0.25
    """Exact identifier matches (a name in the query that a unit defines or uses)."""
    w_path: float = 0.10
    """The query names a directory on the unit's path (1) or its file's stem (0.5):
    "in the billing package", "the webhooks module"."""
    quality_floor: float = 0.5
    """Q(u) and F(u) scale relevance instead of adding to it: S is multiplied by
    ``floor + (1 - floor) * prior``. Added, they gave every candidate about 0.175 for
    free against top scores near 0.45, so weak hits were never far behind."""
    freshness_floor: float = 0.8
    freshness_half_life_days: float = 30.0
    mmr_mu: float = 0.7
    mmr_same_file: float = 0.0
    """MMR redundancy between two units of one file that do not contain each other. A
    unit inside another unit's full text is fully redundant (1.0)."""
    cross_encoder_top: int = 30
    """How many of the top candidates an optional reranker sees."""
    rerank_boost: float = 1.0
    """Added to the relevance of each unit the reranker picks."""
    full_first: int = 10
    """Depth first: the top MMR picks are packed as full code, in order, before the
    knapsack spends what is left. Partial context (a summary where the code was needed)
    cost most answers on the codebase Q&A benchmark. 0 turns it off."""
    full_share: float = 0.75
    """Share of the budget the full-code picks may use; the rest buys breadth."""
    toc_share: float = 0.08
    """Share of the budget kept back for one-line signatures of the picks that did not
    fit otherwise, so the model knows which names exist."""
    min_relative_score: float = 0.5
    """Drop candidates scoring under this fraction of the best score before MMR and
    packing, so weak hits stop padding big budgets."""
    value_signature: float = 0.15
    value_summary: float = 0.35
    """Packing value of a unit's signature and summary relative to its full code. The
    old 0.4 / 0.7 bought ~60 summaries before any code, and prose rarely holds the
    constant or condition a question asks about."""
    lexical_max_df: float = 0.2
    """Query terms in more than this share of the searched units are dropped from the
    lexical query (``job`` and ``queue`` in a job-queue repo)."""
    expand_identifiers: bool = True
    """Append the split parts of identifiers in the query (``TokenBucket`` -> token
    bucket) to the text embedded for the dense views."""

    @property
    def freshness_lambda(self) -> float:
        """λ per day, so F(u) = exp(-λ Δt_days)."""
        return math.log(2) / self.freshness_half_life_days


class WriteBackParams(BaseModel):
    """Sketch §8."""

    dup_threshold: float = 0.92
    min_reference_length: int = 3
    """Shortest identifier that counts as a call or import of a recalled unit in a diff."""


class PagingParams(BaseModel):
    """Sketch §9."""

    high_water: float = 0.70
    low_water: float = 0.50
    session_weight: float = 1.2
    session_half_life_hours: float = 2.0
    keep_recent_turns: int = 2
    """Turns at the end of the window that are never evicted, however full it is."""
    turn_unit_max_tokens: int = 1200
    """Evicted turns are grouped into turn units of at most this size (τ_max, §3)."""


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
