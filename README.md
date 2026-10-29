# CodeBrain

A memory layer for coding agents. Instead of compacting context, CodeBrain pages it
out to an indexed store: every piece of the codebase and every finished agent task
becomes a searchable **unit**, and a new task starts by recalling the code and past
work that look like it.

Design: [CodeBrain design sketch v0.1](https://claude.ai/artifact/Pg4MvYFnt3brZjghyxXUeJ).
Section numbers below (§N) refer to it.

## Quick start

```sh
docker compose up -d db          # Postgres 16 + pgvector on localhost:5432
cp .env.example .env
uv sync
uv run codebrain migrate         # apply schema
CODEBRAIN_DATABASE_URL=postgresql://codebrain:codebrain@localhost:5432/codebrain uv run pytest
```

Tests marked as needing the database skip themselves when `CODEBRAIN_DATABASE_URL`
is unset. They drop and recreate the `public` schema, so point them at a scratch
database.

## Layout

| Module | Sketch | Owns |
| --- | --- | --- |
| `codebrain/models.py` | §2, §6, §7 | Pydantic models for every table: `Unit`, `UnitVector`, `Edge`, `Episode`, `Retrieval`, and the enums (`UnitKind`, `Layer`, `EdgeType`, `VectorView`, `Resolution`, `Namespace`). |
| `codebrain/config.py` | all | `Settings` (env prefix `CODEBRAIN_`) and every tunable constant from the sketch, grouped by section. |
| `codebrain/store/` | §2, §6, §10 | Postgres + pgvector. `migrations/` (SQL), `migrate.py` (runner), `db.py` (connections), `repo.py` (basic reads and writes shared by every writer). |
| `codebrain/ingest/` | §3, §4, §5 | Repo to units: tree-sitter parse, AST chunking within τ bounds, AST hashing, LLM summaries, embeddings, layer and capability classification, call/import/contains edges. *(next)* |
| `codebrain/retrieval/` | §6, §7 | The context broker. `ContextBroker(conn, encoders).recall(query, budget)` and `.explain(query, unit_id)`. Modules: `search` (three indexes, `Filters`), `fusion` (RRF), `graph` (personalized PageRank), `scoring` (S(u\|q) and `ScoreBreakdown`), `mmr`, `pack` (resolution knapsack and rendering), `encoders` (the `Embedder` protocol for query vectors). |
| `codebrain/server/` | §10 | MCP server with `recall(query, k, filters)`, `remember(episode)`, `explain(unit_id)`. *(next)* |
| write-back and paging | §8, §9 | Episode distilling, duplicate merge, usage credit, and window eviction to turn units. Expected to live in `codebrain/memory/` (not created yet). *(next)* |
| `codebrain/cli.py` | | `codebrain migrate`; other commands get added here. |

## Schema

Defined in `src/codebrain/store/migrations/0001_init.sql`.

| Table | Sketch | Notes |
| --- | --- | --- |
| `units` | §2 | One row per unit of any kind. Code ids are `H(repo ‖ path ‖ qualified_name)` (`models.code_unit_id`); experience units get random ids. Holds `validity`, `retrieved_count`, `used_count` for v(u) and Q(u). |
| `units.lexical` | §4 | Generated `tsvector` over `lexical_terms` (split identifiers, weight A) and `summary` (weight B). Stands in for BM25 until a real BM25 is added. |
| `unit_vectors` | §4 | One row per (unit, view, model). Views are `summary` (s_u) and `code` (c_u). The column is untyped `vector` because models are undecided; build an HNSW index per model once one is picked (example in the migration). `content_hash` lets ingest reuse vectors for unchanged code. |
| `edges` | §6 | Typed directed edges. `NULL` weight means the default for the type (`EdgeType.default_weight`). Cascade on unit delete. |
| `episodes` | §8 | Raw evidence (prompt, plan, diff, test results) behind an `episode` unit. |
| `retrievals` | §8 | Every recalled unit with its stage-2 features, for usage credit and learning weights. |

## Working on this in parallel

- **Schema changes** go in a new migration, `NNNN_short_name.sql`, numbered after the
  highest one on `main`. Never edit a merged migration. If two branches pick the same
  number, the second to merge renumbers.
- **New tunables** go in `config.py` under the matching section's params class.
- **Shared writes** (units, vectors, edges, episodes, retrievals) go through
  `store/repo.py`. Search queries live in `retrieval/`, not in `store/`.
- CI runs ruff, `ruff format --check`, mypy (strict) and pytest against
  `pgvector/pgvector:pg16`.

## Recall

```python
from codebrain.retrieval import ContextBroker, Filters
from codebrain.models import VectorView

broker = ContextBroker(conn, {VectorView.SUMMARY: text_embedder, VectorView.CODE: code_embedder})
result = broker.recall(
    "Create a rate limiter for the public API",
    budget=4000,
    filters=Filters(repo="demo", session_id="s1"),
)
print(result.render())  # cited context block, packed to the budget
broker.explain(result.query, "u_...", budget=4000)  # per-feature score breakdown and stage trace
```

An embedder is anything with a `model` id and `embed(texts) -> list[list[float]]`; it
must be the model ingest used for that view. Views without one are skipped, so
`ContextBroker(conn)` gives lexical plus graph recall. `recall` logs what it returns to
`retrievals` (pass `log=False` to skip) inside the caller's transaction.

## Open decisions

Embedding models are not picked yet (§10). `EmbeddingParams` holds placeholders; the
sketch's suggested next step is an eval set of 30 to 50 past "build X" tasks to choose
them by recall@10.
