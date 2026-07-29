// ============================================================================
// src/rate_limiter.rs — RouterFuel v0.7
//
// Tier-based rate limiting for clients, with a real registry backing it.
//
// Previously this only exposed a hardcoded UserTier enum that main.rs called
// with a fixed `UserTier::Pro` for every client — there was no way to
// actually assign different clients different tiers without a redeploy.
// `TierConfig` + `register()`/`status()` here is what `client_registry.rs`
// wires up (from the ROUTERFUEL_CLIENT_TIERS env var and/or the
// `client_tiers` Postgres table), so a client's rate limit now comes from
// that registry, defaulting to `default_tier` for anyone not explicitly
// registered.
// ============================================================================

use governor::{Quota, RateLimiter as GovernorLimiter};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::num::NonZeroU32;
use thiserror::Error;
use tracing::{debug, warn};

#[derive(Error, Debug)]
pub enum RateLimitError {
    #[error("Rate limit exceeded")]
    LimitExceeded,

    #[error("Unknown tier: {0}")]
    UnknownTier(String),
}

// ============================================================================
// TIER CONFIG
// ============================================================================

/// A named rate-limit tier. `capacity` is requests-per-second (also used as
/// the burst size — simple, predictable, and enough for a gateway; swap in
/// a richer governor::Quota if you need separate burst vs. sustained rates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierConfig {
    pub name: &'static str,
    pub capacity: u32,
}

impl TierConfig {
    pub const FREE: TierConfig = TierConfig { name: "free", capacity: 10 };
    pub const PRO: TierConfig = TierConfig { name: "pro", capacity: 100 };
    pub const ENTERPRISE: TierConfig = TierConfig { name: "enterprise", capacity: 1000 };
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ClientStatus {
    pub tier_name: &'static str,
    pub capacity_rps: u32,
}

// ============================================================================
// RATE LIMITER
// ============================================================================

pub struct RateLimiter {
    limiters: RwLock<HashMap<String, governor::DefaultDirectRateLimiter>>,
    /// Which tier each client is currently registered at — this is the
    /// piece that was missing before: a real, mutable, lookupable registry.
    tiers: RwLock<HashMap<String, TierConfig>>,
    default_tier: RwLock<TierConfig>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limiters: RwLock::new(HashMap::new()),
            tiers: RwLock::new(HashMap::new()),
            default_tier: RwLock::new(TierConfig::PRO),
        }
    }

    pub fn with_default_tier(self, tier: TierConfig) -> Self {
        *self.default_tier.write() = tier;
        self
    }

    pub fn set_default_tier(&self, tier: TierConfig) {
        *self.default_tier.write() = tier;
    }

    /// Register (or re-register) a client at a specific tier. Safe to call
    /// repeatedly — e.g. client_registry.rs re-syncing from the DB — since
    /// it just replaces the client's limiter at the new capacity.
    pub fn register(&self, client_id: &str, tier: TierConfig) {
        let quota = Quota::per_second(
            NonZeroU32::new(tier.capacity.max(1)).unwrap(),
        );
        self.limiters
            .write()
            .insert(client_id.to_string(), GovernorLimiter::direct(quota));
        self.tiers.write().insert(client_id.to_string(), tier);

        debug!(client_id, tier = tier.name, capacity = tier.capacity, "Registered client tier");
    }

    /// Check whether `client_id` may make a request right now, using
    /// whatever tier they're registered at (or `default_tier` if they've
    /// never been explicitly registered — auto-registering them at that
    /// tier so subsequent lookups are consistent).
    pub fn check(&self, client_id: &str) -> Result<(), RateLimitError> {
        let tier = self.tiers.read().get(client_id).copied();

        let tier = match tier {
            Some(t) => t,
            None => {
                let default = *self.default_tier.read();
                self.register(client_id, default);
                default
            }
        };

        self.check_limit(client_id, tier)
    }

    /// Lower-level check against an explicit tier, bypassing the registry.
    /// Kept for callers that already know the tier they want to enforce.
    pub fn check_limit(&self, client_id: &str, tier: TierConfig) -> Result<(), RateLimitError> {
        let mut limiters = self.limiters.write();

        let limiter = limiters
            .entry(client_id.to_string())
            .or_insert_with(|| {
                GovernorLimiter::direct(Quota::per_second(NonZeroU32::new(tier.capacity.max(1)).unwrap()))
            });

        if limiter.check().is_err() {
            warn!(client_id, tier = tier.name, "Rate limit exceeded");
            return Err(RateLimitError::LimitExceeded);
        }

        Ok(())
    }

    /// What tier is this client currently registered at? Used by the admin
    /// dashboard and by client_registry.rs's own tests.
    pub fn status(&self, client_id: &str) -> Option<ClientStatus> {
        self.tiers.read().get(client_id).map(|t| ClientStatus {
            tier_name: t.name,
            capacity_rps: t.capacity,
        })
    }

    pub fn registered_count(&self) -> usize {
        self.tiers.read().len()
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_capacities() {
        assert_eq!(TierConfig::FREE.capacity, 10);
        assert_eq!(TierConfig::PRO.capacity, 100);
        assert_eq!(TierConfig::ENTERPRISE.capacity, 1000);
    }

    #[test]
    fn test_register_and_status() {
        let rl = RateLimiter::new();
        rl.register("client-a", TierConfig::ENTERPRISE);
        let status = rl.status("client-a").unwrap();
        assert_eq!(status.tier_name, "enterprise");
        assert_eq!(status.capacity_rps, 1000);
    }

    #[test]
    fn test_unregistered_client_gets_default_tier_on_check() {
        let rl = RateLimiter::new().with_default_tier(TierConfig::FREE);
        assert!(rl.status("brand-new-client").is_none());
        assert!(rl.check("brand-new-client").is_ok());
        assert_eq!(rl.status("brand-new-client").unwrap().tier_name, "free");
    }

    #[test]
    fn test_basic_check_limit() {
        let limiter = RateLimiter::new();
        let result = limiter.check_limit("client-1", TierConfig::FREE);
        assert!(result.is_ok());
    }
}
