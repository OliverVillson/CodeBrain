-- 0001: units, vectors, edges, episodes, retrieval log.
-- Mirrors codebrain.models. See README "Schema" for how each table maps to the sketch.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE unit_kind AS ENUM
    ('function', 'method', 'class', 'module', 'prompt', 'episode', 'decision', 'turn');
CREATE TYPE unit_namespace AS ENUM ('code', 'project', 'session');
CREATE TYPE unit_layer AS ENUM ('ui', 'api', 'domain', 'data', 'infra', 'test', 'tooling');
CREATE TYPE edge_type AS ENUM
    ('calls', 'imports', 'contains', 'tests', 'produced_by', 'similar_to');
CREATE TYPE vector_view AS ENUM ('summary', 'code');
CREATE TYPE unit_resolution AS ENUM ('signature', 'summary', 'full');

-- One row per memory unit u (sketch §2).
CREATE TABLE units (
    id               text PRIMARY KEY,
    kind             unit_kind NOT NULL,
    namespace        unit_namespace NOT NULL DEFAULT 'code',
    repo             text,
    session_id       text,

    path             text,
    qualified_name   text,
    language         text,
    start_line       integer CHECK (start_line >= 1),
    end_line         integer CHECK (end_line >= start_line),
    parent_id        text REFERENCES units (id) ON DELETE SET NULL,

    text             text NOT NULL,
    summary          text,
    signature        text,
    -- Split identifiers (getUserById -> get user by id), written by ingest.
    lexical_terms    text,
    -- Lexical view ℓ_u. ts_rank_cd stands in for BM25 until a real BM25 lands.
    lexical          tsvector GENERATED ALWAYS AS (
                         setweight(to_tsvector('simple', coalesce(lexical_terms, '')), 'A') ||
                         setweight(to_tsvector('english', coalesce(summary, '')), 'B')
                     ) STORED,

    ast_hash         text,
    tokens           integer NOT NULL DEFAULT 0 CHECK (tokens >= 0),

    layer            unit_layer,
    layer_confidence real CHECK (layer_confidence BETWEEN 0 AND 1),
    capabilities     text[] NOT NULL DEFAULT '{}',
    labels           jsonb NOT NULL DEFAULT '{}',

    -- v(u): 1 current, 0.3 changed or gone, 0 deprecated (sketch §7).
    validity         real NOT NULL DEFAULT 1 CHECK (validity BETWEEN 0 AND 1),
    retrieved_count  integer NOT NULL DEFAULT 0 CHECK (retrieved_count >= 0),
    used_count       integer NOT NULL DEFAULT 0 CHECK (used_count >= 0),
    version          integer NOT NULL DEFAULT 1 CHECK (version >= 1),

    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),

    CHECK (namespace <> 'session' OR session_id IS NOT NULL),
    CHECK (kind NOT IN ('function', 'method', 'class', 'module', 'prompt') OR path IS NOT NULL)
);

CREATE INDEX units_kind_idx      ON units (kind);
CREATE INDEX units_scope_idx     ON units (namespace, repo);
CREATE INDEX units_session_idx   ON units (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX units_path_idx      ON units (repo, path);
CREATE INDEX units_parent_idx    ON units (parent_id);
CREATE INDEX units_ast_hash_idx  ON units (ast_hash);
CREATE INDEX units_layer_idx     ON units (layer);
CREATE INDEX units_capabilities_idx ON units USING gin (capabilities);
CREATE INDEX units_lexical_idx   ON units USING gin (lexical);

-- Dense views s_u and c_u (sketch §4). Dimension is left open because the models
-- are undecided; an ANN index is per (view, model), built once a model is chosen:
--   CREATE INDEX ON unit_vectors USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
--       WHERE view = 'summary' AND model = '<model id>';
CREATE TABLE unit_vectors (
    unit_id       text NOT NULL REFERENCES units (id) ON DELETE CASCADE,
    view          vector_view NOT NULL,
    model         text NOT NULL,
    dim           integer NOT NULL CHECK (dim > 0),
    embedding     vector NOT NULL CHECK (vector_dims(embedding) = dim),
    -- Hash of the embedded input; ingest reuses vectors when it is unchanged.
    content_hash  text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (unit_id, view, model)
);

CREATE INDEX unit_vectors_content_hash_idx ON unit_vectors (model, view, content_hash);

-- Code graph G = (V, E) (sketch §6). NULL weight means the edge type's default.
CREATE TABLE edges (
    src_id     text NOT NULL REFERENCES units (id) ON DELETE CASCADE,
    dst_id     text NOT NULL REFERENCES units (id) ON DELETE CASCADE,
    type       edge_type NOT NULL,
    weight     real CHECK (weight >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (src_id, dst_id, type)
);

CREATE INDEX edges_dst_idx ON edges (dst_id, type);

-- Raw evidence for an episode unit (sketch §8).
CREATE TABLE episodes (
    unit_id       text PRIMARY KEY REFERENCES units (id) ON DELETE CASCADE,
    session_id    text,
    prompt        text NOT NULL,
    plan          text,
    diff          text,
    test_results  jsonb NOT NULL DEFAULT '{}',
    tests_passed  boolean,
    pitfalls      text[] NOT NULL DEFAULT '{}',
    started_at    timestamptz,
    ended_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX episodes_session_idx ON episodes (session_id);

-- One row per unit returned by recall: usage credit and the training set for
-- learning the stage-2 weights (sketch §8).
CREATE TABLE retrievals (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    query       text NOT NULL,
    unit_id     text NOT NULL REFERENCES units (id) ON DELETE CASCADE,
    session_id  text,
    rank        integer NOT NULL CHECK (rank >= 1),
    score       real NOT NULL,
    resolution  unit_resolution NOT NULL,
    features    jsonb NOT NULL DEFAULT '{}',
    used        boolean,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX retrievals_unit_idx    ON retrievals (unit_id);
CREATE INDEX retrievals_session_idx ON retrievals (session_id, created_at);
