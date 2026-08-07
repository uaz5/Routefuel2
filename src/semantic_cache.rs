// =============================================================================
// src/semantic_cache.rs  — RouterFuel v0.7 (fixed)
//
// Semantic Cache using pgvector + a local ONNX embedding model.
//
// FIX (this revision): store()'s prompt_preview truncation used to slice
// the prompt by raw byte index (`&prompt[..prompt.len().min(200)]`), which
// is not UTF-8 char-boundary-safe. Any prompt containing multi-byte
// characters (accented letters, CJK, emoji, etc.) whose 200th byte landed
// mid-character would panic ("byte index 200 is not a char boundary")
// inside the tokio::spawn'd store task — silently and repeatedly breaking
// caching for that prompt shape. Now truncates on a char boundary via
// `prompt.chars().take(200).collect()`.
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

    #[instrument(skip(self, prompt))]
   pub async fn lookup(&self, prompt: &str, model: &str) -> Option<CacheHit> {
        if !self.enabled { return None; }

        let start = Instant::now();
        let hash = sha256_hex(&format!("{model}\u{0}{prompt}"));

        let exact: Option<(String, String)> = sqlx::query_as(
            "SELECT cached_response, model_used
             FROM semantic_cache
             WHERE prompt_hash = $1
             LIMIT 1",
        )
        .bind(&hash)
        .fetch_optional(self.pool.as_ref())
        .await
        .ok()
        .flatten();

        if let Some((response, model)) = exact {
            debug!(latency_us = start.elapsed().as_micros(), "Exact cache hit");
            let pool = Arc::clone(&self.pool);
            let hash_clone = hash.clone();
            tokio::spawn(async move {
                let _ = sqlx::query(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1,
                     last_hit_at = NOW() WHERE prompt_hash = $1"
                )
                .bind(&hash_clone)
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
       AND 1 - (embedding <=> $1::vector) >= $2
     ORDER BY embedding <=> $1::vector
     LIMIT 1",
)
.bind(&vector)
.bind(SIMILARITY_THRESHOLD)
.bind(model)
.fetch_optional(self.pool.as_ref())
.await
.ok()
.flatten();

if let Some((response, model, similarity, matched_hash)) = row {
    debug!(similarity = similarity, latency_ms = start.elapsed().as_millis(), "Semantic cache hit");

    let pool = Arc::clone(&self.pool);
    tokio::spawn(async move {
        let _ = sqlx::query(
            "UPDATE semantic_cache
             SET hit_count = hit_count + 1, last_hit_at = NOW()
             WHERE prompt_hash = $1"
        )
        .bind(&matched_hash)
        .execute(pool.as_ref())
        .await;
    });

    return Some(CacheHit { cached_response: response, model_used: model, similarity });
}
        debug!(latency_ms = start.elapsed().as_millis(), "Cache miss");
        None
    }

    pub fn store(&self, prompt: String, cached_response: String, model_used: String) {
        if !self.enabled { return; }

        let Some(embedder) = self.embedder.clone() else { return; };
        let pool = Arc::clone(&self.pool);

        tokio::spawn(async move {
            let hash = sha256_hex(&format!("{model_used}\u{0}{prompt}"));

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

            // FIX: was `&prompt[..prompt.len().min(200)]` — a raw byte-index
            // slice, not char-boundary-safe. Panicked on any prompt where a
            // multi-byte UTF-8 character straddled byte 200. Truncating by
            // `.chars()` instead guarantees a valid boundary regardless of
            // the prompt's script/encoding.
            let preview: String = prompt.chars().take(200).collect();

            if let Err(e) = sqlx::query(
                "INSERT INTO semantic_cache
                    (prompt_hash, prompt_preview, embedding, cached_response, model_used)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (prompt_hash) DO NOTHING",
            )
            .bind(&hash)
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

fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::EMBEDDING_DIMS;

    #[test]
    fn embedding_dims_matches_migration_005() {
        // Keeps this constant honest against migrations/005_local_embeddings.sql,
        // which declares `embedding vector(384)`.
        assert_eq!(EMBEDDING_DIMS, 384);
    }
}
