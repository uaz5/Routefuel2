// =============================================================================
// src/semantic_cache.rs  — RouterFuel v0.8 (fixed)
//
// Semantic Cache using pgvector + a local ONNX embedding model.
//
// FIX (this revision): lookup()/store() previously keyed purely on
// (model, prompt-hash / embedding similarity) with NO client scoping at
// all. In a multi-tenant BYOK gateway that meant Client A's cached
// response could be served verbatim to Client B on an identical or
// semantically-similar prompt — a cross-tenant confidentiality leak, and
// also a way for a client with no working BYOK key to free-ride on another
// tenant's cached completions. Every lookup/store now takes a
// `client_id: &str` and includes it in the prompt hash and both the exact
// and vector-similarity queries, so a client can only ever hit its own
// cache entries. Requires the companion migration adding
// `client_id VARCHAR(64) NOT NULL` to `semantic_cache` (see
// 007_semantic_cache_client_scope.sql) and dropping the old
// `UNIQUE(prompt_hash)` in favor of `UNIQUE(prompt_hash, client_id)`.
//
// Anonymous/no-client-id traffic (if you ever allow it) should pass a
// stable sentinel like "anonymous" — same convention main.rs already uses
// for rate limiting/spend guarding — rather than an empty string, so it
// still gets its own isolated cache partition instead of colliding with a
// real client_id.
//
// FIX (carried over from prior revision): store()'s prompt_preview
// truncation is char-boundary-safe via `.chars().take(200).collect()`
// rather than a raw byte-index slice, which used to panic on any prompt
// containing multi-byte UTF-8 characters landing on byte 200.
// =============================================================================

use crate::embedder::{LocalEmbedder, EMBEDDING_DIMS};
use anyhow::Result;
use pgvector::Vector;
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, instrument};

const SIMILARITY_THRESHOLD: f64 = 0.96;

#[derive(Debug, Clone)]
pub struct CacheHit {
    pub cached_response: String,
    pub model_used: String,
    pub similarity: f64,
}

pub struct SemanticCache {
    pool: Arc<PgPool>,
    embedder: Option<Arc<LocalEmbedder>>,
    enabled: bool,
}

impl SemanticCache {
    pub fn new(pool: Arc<PgPool>, embedder: Option<Arc<LocalEmbedder>>) -> Self {
        let enabled = embedder.is_some();
        Self { pool, embedder, enabled }
    }

    pub fn disable(&mut self) { self.enabled = false; }

    /// `client_id` scopes this lookup to one tenant's cache partition —
    /// see the FIX note at the top of this file. Callers in main.rs should
    /// pass the same rl_key/client_hash used for rate limiting and spend
    /// guarding (falling back to "anonymous" for unauthenticated-but-
    /// permitted traffic, never an empty string).
    #[instrument(skip(self, prompt))]
    pub async fn lookup(&self, client_id: &str, prompt: &str, model: &str) -> Option<CacheHit> {
        if !self.enabled { return None; }

        let start = Instant::now();
        let hash = cache_key_hash(client_id, model, prompt);

        let exact: Option<(String, String)> = sqlx::query_as(
            "SELECT cached_response, model_used
             FROM semantic_cache
             WHERE prompt_hash = $1 AND client_id = $2
             LIMIT 1",
        )
        .bind(&hash)
        .bind(client_id)
        .fetch_optional(self.pool.as_ref())
        .await
        .ok()
        .flatten();

        if let Some((response, model)) = exact {
            debug!(latency_us = start.elapsed().as_micros(), "Exact cache hit");
            let pool = Arc::clone(&self.pool);
            let hash_clone = hash.clone();
            let client_id_clone = client_id.to_string();
            tokio::spawn(async move {
                let _ = sqlx::query(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1,
                     last_hit_at = NOW() WHERE prompt_hash = $1 AND client_id = $2"
                )
                .bind(&hash_clone)
                .bind(&client_id_clone)
                .execute(pool.as_ref())
                .await;
            });
            return Some(CacheHit {
                cached_response: response,
                model_used: model,
                similarity: 1.0,
            });
        }

        let embedding_vec = match self.embed(prompt) {
            Ok(v) => v,
            Err(e) => {
                debug!("Local embedding failed: {e}");
                return None;
            }
        };

        let vector = Vector::from(embedding_vec);

        let row: Option<(String, String, f64, String)> = sqlx::query_as(
            "SELECT cached_response, model_used,
                    1 - (embedding <=> $1::vector) AS similarity,
                    prompt_hash
             FROM semantic_cache
             WHERE model_used = $3
               AND client_id = $4
               AND 1 - (embedding <=> $1::vector) >= $2
             ORDER BY embedding <=> $1::vector
             LIMIT 1",
        )
        .bind(&vector)
        .bind(SIMILARITY_THRESHOLD)
        .bind(model)
        .bind(client_id)
        .fetch_optional(self.pool.as_ref())
        .await
        .ok()
        .flatten();

        if let Some((response, model, similarity, matched_hash)) = row {
            debug!(similarity = similarity, latency_ms = start.elapsed().as_millis(), "Semantic cache hit");

            let pool = Arc::clone(&self.pool);
            let client_id_clone = client_id.to_string();
            tokio::spawn(async move {
                let _ = sqlx::query(
                    "UPDATE semantic_cache
                     SET hit_count = hit_count + 1, last_hit_at = NOW()
                     WHERE prompt_hash = $1 AND client_id = $2"
                )
                .bind(&matched_hash)
                .bind(&client_id_clone)
                .execute(pool.as_ref())
                .await;
            });

            return Some(CacheHit { cached_response: response, model_used: model, similarity });
        }
        debug!(latency_ms = start.elapsed().as_millis(), "Cache miss");
        None
    }

    /// `client_id` scopes this write to one tenant's cache partition — see
    /// the FIX note at the top of this file.
    pub fn store(&self, client_id: String, prompt: String, cached_response: String, model_used: String) {
        if !self.enabled { return; }

        let Some(embedder) = self.embedder.clone() else { return; };
        let pool = Arc::clone(&self.pool);

        tokio::spawn(async move {
            let hash = cache_key_hash(&client_id, &model_used, &prompt);

            let embed_result = {
                let prompt_clone = prompt.clone();
                tokio::task::spawn_blocking(move || embedder.embed(&prompt_clone)).await
            };

            let embedding = match embed_result {
                Ok(Ok(e)) => e,
                Ok(Err(e)) => {
                    debug!("Cache store embedding failed: {e}");
                    return;
                }
                Err(e) => {
                    debug!("Cache store embedding task panicked: {e}");
                    return;
                }
            };

            let vector = Vector::from(embedding);

            // Char-boundary-safe truncation — see FIX note at top of file.
            let preview: String = prompt.chars().take(200).collect();

            if let Err(e) = sqlx::query(
                "INSERT INTO semantic_cache
                    (prompt_hash, client_id, prompt_preview, embedding, cached_response, model_used)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 ON CONFLICT (prompt_hash, client_id) DO NOTHING",
            )
            .bind(&hash)
            .bind(&client_id)
            .bind(&preview)
            .bind(&vector)
            .bind(&cached_response)
            .bind(&model_used)
            .execute(pool.as_ref())
            .await
            {
                debug!("Cache store failed: {e}");
            } else {
                debug!("Cached response for prompt (hash: {})", &hash[..8]);
            }
        });
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match &self.embedder {
            Some(e) => e.embed(text),
            None => Err(anyhow::anyhow!("no embedder")),
        }
    }
}

/// Hash now includes client_id as well as model — two different clients
/// sending the byte-identical prompt to the byte-identical model must land
/// in different cache partitions. NUL-separated to avoid any ambiguity
/// between e.g. client_id="ab" + model="c" and client_id="a" + model="bc".
fn cache_key_hash(client_id: &str, model: &str, prompt: &str) -> String {
    let mut h = Sha256::new();
    h.update(client_id.as_bytes());
    h.update(b"\0");
    h.update(model.as_bytes());
    h.update(b"\0");
    h.update(prompt.as_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_dims_matches_migration_005() {
        assert_eq!(EMBEDDING_DIMS, 384);
    }

    #[test]
    fn cache_key_differs_by_client() {
        let a = cache_key_hash("client-a", "claude-sonnet-5", "hello");
        let b = cache_key_hash("client-b", "claude-sonnet-5", "hello");
        assert_ne!(a, b, "different clients must not share a cache key for the same prompt+model");
    }

    #[test]
    fn cache_key_matches_for_same_client_prompt_model() {
        let a = cache_key_hash("client-a", "claude-sonnet-5", "hello");
        let b = cache_key_hash("client-a", "claude-sonnet-5", "hello");
        assert_eq!(a, b);
    }
}
