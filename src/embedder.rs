// =============================================================================
// src/embedder.rs — RouterFuel v0.9
//
// FIX (this revision): embed() used to treat a poisoned mutex as a hard,
// permanent error — `.lock().map_err(...)`. A single panic inside
// ort::Session::run() (FFI into ONNX Runtime) would poison the mutex
// forever, silently disabling semantic caching for the rest of the
// process's life with no recovery path. Now recovers via
// `poisoned.into_inner()` — the session itself is still usable after a
// failed run, only the lock's poison flag needs clearing.
// =============================================================================

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument, warn};

pub const EMBEDDING_DIMS: usize = 384;

pub struct LocalEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl LocalEmbedder {
    #[instrument]
    pub fn load(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("failed to create ONNX Runtime session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("failed to set graph optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow!("failed to set intra-op thread count: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("failed to load ONNX model from '{}': {}", model_path, e))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer from '{}': {}", tokenizer_path, e))?;

        info!(model_path, tokenizer_path, dims = EMBEDDING_DIMS, "Loaded local embedding model");

        Ok(Self { session: Mutex::new(session), tokenizer })
    }

    #[instrument(skip(self, text))]
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer encode failed: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq_len = ids.len();

        if seq_len == 0 {
            return Ok(vec![0.0; EMBEDDING_DIMS]);
        }

        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], ids))
            .context("failed to build input_ids tensor")?;
        let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], mask.clone()))
            .context("failed to build attention_mask tensor")?;
        let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], type_ids))
            .context("failed to build token_type_ids tensor")?;

        // FIX: was `.lock().map_err(|_| anyhow!("... poisoned ..."))?` —
        // a permanent hard failure once poisoned. Recover the guard
        // instead: the underlying Session is still valid to keep using
        // after a single failed run; only the lock's poison flag needs
        // clearing so future calls aren't blocked forever by one panic.
        let mut session = match self.session.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    "embedding session mutex was poisoned by a previous panic — \
                     recovering and continuing rather than permanently disabling \
                     semantic caching"
                );
                poisoned.into_inner()
            }
        };

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .context("ONNX Runtime inference failed")?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("failed to extract last_hidden_state tensor")?;

        let hidden_dim = *shape.last().ok_or_else(|| anyhow!("unexpected empty tensor shape"))? as usize;

        let mut pooled = vec![0f32; hidden_dim];
        let mut valid_tokens = 0f32;

        for (pos, &m) in mask.iter().enumerate() {
            if m != 1 {
                continue;
            }
            let base = pos * hidden_dim;
            for d in 0..hidden_dim {
                pooled[d] += data[base + d];
            }
            valid_tokens += 1.0;
        }

        if valid_tokens > 0.0 {
            for v in pooled.iter_mut() {
                *v /= valid_tokens;
            }
        }

        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in pooled.iter_mut() {
                *v /= norm;
            }
        }

        if pooled.len() != EMBEDDING_DIMS {
            debug!(
                actual_dims = pooled.len(),
                expected_dims = EMBEDDING_DIMS,
                "Embedding model output dimension does not match EMBEDDING_DIMS constant"
            );
        }

        Ok(pooled)
    }
}
