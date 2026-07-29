// =============================================================================
// src/client_registry.rs  — RouterFuel v0.7
//
// Per-client tier assignment — this is what closes the TODO left in
// main.rs's rate-limit check ("wire this up to the client_tiers table for
// per-client overrides"). Loaded at startup from either:
//
//   A) Environment variable ROUTERFUEL_CLIENT_TIERS (fast, no DB needed)
//      Format:  "raw_key_1:pro,raw_key_2:enterprise,raw_key_3:free"
//      Keys are hashed to match ApiKeyStore's client_id convention.
//
//   B) Postgres table `client_tiers` (for runtime changes without redeploy —
//      the table itself already exists via migrations/003_client_tiers.sql)
//
// Tier changes made only in the DB take effect on the next restart unless
// you also call `load_all_tiers` on a timer — that's a reasonable follow-up
// if you want live updates without a redeploy, not included here to avoid
// adding a background task nobody asked for yet.
// =============================================================================

use crate::rate_limiter::{RateLimiter, TierConfig};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use tracing::{error, info, warn};

// =============================================================================
// Tier parsing
// =============================================================================

pub fn parse_tier(s: &str) -> TierConfig {
    match s.trim().to_lowercase().as_str() {
        "free"       => TierConfig::FREE,
        "pro"        => TierConfig::PRO,
        "enterprise" => TierConfig::ENTERPRISE,
        other => {
            warn!("Unknown tier '{}' — defaulting to Pro", other);
            TierConfig::PRO
        }
    }
}

// =============================================================================
// Load from environment variable
//
// ROUTERFUEL_CLIENT_TIERS format:
//   "raw_key_1:pro,raw_key_2:enterprise,raw_key_3:free"
//
// Keys are stored as SHA-256 hashes in ApiKeyStore (see auth.rs) — here we
// accept the raw key so the same secret works for both auth and tier
// assignment, and hash it the same way ApiKeyStore does.
// =============================================================================

pub fn load_tiers_from_env(raw: &str, rate_limiter: &Arc<RateLimiter>) -> usize {
    let mut count = 0;

    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() { continue; }

        match entry.split_once(':') {
            Some((key, tier_str)) => {
                let client_id = sha256_hex(key.trim());
                let tier = parse_tier(tier_str);

                rate_limiter.register(&client_id, tier);

                info!(
                    client_id = &client_id[..8],
                    tier = tier_str.trim(),
                    "Registered client tier from env"
                );
                count += 1;
            }
            None => {
                error!(
                    "Bad ROUTERFUEL_CLIENT_TIERS entry '{}' — format: raw_key:tier",
                    entry
                );
            }
        }
    }

    count
}

// =============================================================================
// Load from Postgres (optional — only if the client_tiers table exists;
// see migrations/003_client_tiers.sql, which already creates it)
// =============================================================================

pub async fn load_tiers_from_db(
    pool: &PgPool,
    rate_limiter: &Arc<RateLimiter>,
) -> Result<usize, sqlx::Error> {
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'client_tiers'
         )"
    )
    .fetch_one(pool)
    .await?;

    if !table_exists {
        info!("client_tiers table not found — skipping DB tier load");
        return Ok(0);
    }

    let rows = sqlx::query_as::<_, (String, String)>(
        "SELECT client_id, tier FROM client_tiers ORDER BY updated_at DESC"
    )
    .fetch_all(pool)
    .await?;

    let count = rows.len();

    for (client_id, tier_str) in rows {
        let tier = parse_tier(&tier_str);
        rate_limiter.register(&client_id, tier);
        info!(
            client_id = &client_id[..client_id.len().min(8)],
            tier = %tier_str,
            "Registered client tier from DB"
        );
    }

    info!("Loaded {} client tiers from database", count);
    Ok(count)
}

// =============================================================================
// Combined loader — env first, then DB (DB entries are registered after, so
// if the same client_id appears in both, the DB value is what's active —
// intentional, since the DB is the one you can change without a redeploy).
// =============================================================================

pub async fn load_all_tiers(
    pool: &PgPool,
    rate_limiter: &Arc<RateLimiter>,
    env_tiers_raw: &str,
    default_tier: TierConfig,
) {
    rate_limiter.set_default_tier(default_tier);

    let mut total = 0;

    if !env_tiers_raw.is_empty() {
        let n = load_tiers_from_env(env_tiers_raw, rate_limiter);
        total += n;
        info!("Loaded {} client tiers from ROUTERFUEL_CLIENT_TIERS", n);
    }

    match load_tiers_from_db(pool, rate_limiter).await {
        Ok(n) => {
            total += n;
            info!("Loaded {} client tiers from database", n);
        }
        Err(e) => {
            warn!("Could not load tiers from DB ({}), using env only", e);
        }
    }

    if total == 0 {
        warn!(
            "No client tiers configured — every client will get the '{}' tier ({} req/s) \
             until explicitly registered. Set ROUTERFUEL_CLIENT_TIERS or add rows to \
             client_tiers.",
            default_tier.name, default_tier.capacity
        );
    }

    info!("Client tier registry ready ({} entries, default = {})", total, default_tier.name);
}

// =============================================================================
// Helper
// =============================================================================

fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    format!("{:x}", h.finalize())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_tiers() {
        assert_eq!(parse_tier("free").capacity, TierConfig::FREE.capacity);
        assert_eq!(parse_tier("pro").capacity, TierConfig::PRO.capacity);
        assert_eq!(parse_tier("enterprise").capacity, TierConfig::ENTERPRISE.capacity);
    }

    #[test]
    fn parse_unknown_defaults_to_pro() {
        let t = parse_tier("gold");
        assert_eq!(t.capacity, TierConfig::PRO.capacity);
    }

    #[test]
    fn load_from_env_registers_clients() {
        let rl = Arc::new(RateLimiter::new());
        let raw = "rf_live_key1:pro,rf_live_key2:free,rf_live_key3:enterprise";
        let n = load_tiers_from_env(raw, &rl);
        assert_eq!(n, 3);

        let id1 = sha256_hex("rf_live_key1");
        assert!(rl.status(&id1).is_some());
        assert_eq!(rl.status(&id1).unwrap().tier_name, "pro");
    }

    #[test]
    fn empty_env_string_registers_nothing() {
        let rl = Arc::new(RateLimiter::new());
        let n = load_tiers_from_env("", &rl);
        assert_eq!(n, 0);
    }
}
