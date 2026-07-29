-- =============================================================================
-- migrations/005_local_embeddings.sql  — RouterFuel v0.6
--
-- Switches the semantic cache from OpenAI's paid text-embedding-3-small
-- (1536 dims, billed to RouterFuel's own key — a BYOK violation) to a local
-- ONNX sentence-embedding model (384 dims, e.g. all-MiniLM-L6-v2) run
-- in-process by src/embedder.rs. Zero external cost, no network round trip.
--
-- pgvector cannot resize a vector column in place, and old 1536-dim
-- embeddings are meaningless to compare against new 384-dim ones anyway, so
-- this migration truncates the cache (cached *responses* are not lost
-- anywhere else — this table is purely a performance cache, safe to empty)
-- and rebuilds the column + index at the new dimension.
--
-- The cosine-similarity hit threshold (0.96) is defined in
-- src/semantic_cache.rs and is NOT changed by this migration.
-- =============================================================================

TRUNCATE TABLE semantic_cache;

DROP INDEX IF EXISTS idx_semantic_cache_embedding;

ALTER TABLE semantic_cache
    ALTER COLUMN embedding TYPE vector(384);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding
    ON semantic_cache
    USING hnsw (embedding vector_cosine_ops);

COMMENT ON COLUMN semantic_cache.embedding IS
    '384-dim float32 vector from a locally-run ONNX sentence-embedding model (see src/embedder.rs). Not billed to any provider.';
