-- =============================================================================
-- migrations/002_semantic_cache_and_vision.sql  — RouteFuel v0.5
-- Adds:
--   1. pgvector extension
--   2. semantic_cache table with HNSW index (1536-dim for OpenAI text-embedding-3-small)
--   3. from_cache column on request_logs
--   4. vision_requests table for image metadata
-- =============================================================================

-- ── 1. Enable pgvector ────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── 2. Semantic cache table ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS semantic_cache (
    id                  BIGSERIAL       PRIMARY KEY,

    -- Exact SHA-256 hash of the full prompt text (for free O(1) exact lookup)
    prompt_hash         VARCHAR(64)     NOT NULL UNIQUE,

    -- Short preview for dashboard display (never store full prompt for privacy)
    prompt_preview      VARCHAR(200),

    -- 1536-dim vector from OpenAI text-embedding-3-small model
    embedding           vector(1536)    NOT NULL,

    -- The full ChatResponse JSON we return on a cache hit
    cached_response     TEXT            NOT NULL,

    -- Which model originally generated this response
    model_used          VARCHAR(100)    NOT NULL,

    -- Analytics
    hit_count           INTEGER         NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_hit_at         TIMESTAMPTZ
);

-- Index for exact hash lookups (super fast)
CREATE INDEX IF NOT EXISTS idx_semantic_cache_hash 
    ON semantic_cache(prompt_hash);

-- Index for vector approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding
    ON semantic_cache
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_hit_count
    ON semantic_cache (hit_count DESC);

COMMENT ON TABLE  semantic_cache             IS 'Stores prompt embeddings + responses for semantic similarity caching';
COMMENT ON COLUMN semantic_cache.embedding   IS '1536-dim float32 vector from OpenAI text-embedding-3-small model';
COMMENT ON COLUMN semantic_cache.hit_count   IS 'Number of times this entry was returned as a cache hit';

-- ── 3. Add from_cache flag to request_logs ────────────────────────────────────
ALTER TABLE request_logs
    ADD COLUMN IF NOT EXISTS from_cache BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_rl_from_cache
    ON request_logs (from_cache)
    WHERE from_cache = TRUE;

COMMENT ON COLUMN request_logs.from_cache IS 'TRUE if response was served from semantic cache (no LLM call made)';

-- ── 4. Vision requests metadata ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vision_requests (
    id                  BIGSERIAL       PRIMARY KEY,
    request_id          VARCHAR(36)     NOT NULL REFERENCES request_logs(request_id) ON DELETE CASCADE,
    image_count         SMALLINT        NOT NULL DEFAULT 1,
    image_source        VARCHAR(20),
    media_type          VARCHAR(50),
    detail_level        VARCHAR(10),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vision_request_id
    ON vision_requests (request_id);

COMMENT ON TABLE vision_requests IS 'Image metadata for multimodal requests — joined to request_logs';