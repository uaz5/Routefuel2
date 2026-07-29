// =============================================================================
// src/guardrails.rs — RouterFuel v0.7
//
// Two independent safety nets against a runaway agent quietly racking up a
// huge bill (the client's bill, since RouterFuel is BYOK-only — but a client
// stuck in a loop is still a support fire, a trust problem, and exactly the
// kind of thing a gateway should catch before the 500th identical call, not
// after). Both checks run BEFORE any provider is called, so a blocked
// request costs nothing anywhere.
//
//   LoopGuard  — flags a client sending the same prompt over and over in a
//                short window. This is the single most common signature of
//                an agent stuck retrying the same failed step, or two agents
//                bouncing a message back and forth.
//   SpendGuard — hard per-client ceiling on cost within a rolling window,
//                independent of *why* the spend happened. Catches loops that
//                vary their prompt each time (so LoopGuard alone wouldn't
//                catch them) as well as simple runaway volume.
//
// Both are in-memory (DashMap) and process-local by design: they need to be
// sub-millisecond on the hot path, so no DB round trip. That means limits
// reset if the process restarts and aren't shared across horizontally-scaled
// replicas — fine for a loop/spend *guard*, not a substitute for the
// authoritative accounting in cost_tracker.rs / request_logs.
// =============================================================================

use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use tracing::warn;

// ============================================================================
// LOOP GUARD
// ============================================================================

/// A client repeating the exact same prompt this many times inside
/// LOOP_WINDOW is treated as a runaway loop, not legitimate traffic.
const DEFAULT_LOOP_REPEAT_THRESHOLD: usize = 4;
const DEFAULT_LOOP_WINDOW: Duration = Duration::from_secs(60);
/// Cap on how much history we keep per client so a very chatty legitimate
/// client can't grow this map unbounded.
const MAX_HISTORY_PER_CLIENT: usize = 200;

pub struct LoopGuard {
    recent: DashMap<String, VecDeque<(u64, Instant)>>,
    threshold: usize,
    window: Duration,
}

impl LoopGuard {
    pub fn new() -> Self {
        Self::with_config(DEFAULT_LOOP_REPEAT_THRESHOLD, DEFAULT_LOOP_WINDOW)
    }

    pub fn with_config(threshold: usize, window: Duration) -> Self {
        Self { recent: DashMap::new(), threshold, window }
    }

    /// Records this call and returns `true` if it looks like a runaway loop
    /// (the same prompt hash has now appeared `threshold` or more times for
    /// this client inside `window`).
    pub fn check_and_record(&self, client_id: &str, prompt: &str) -> bool {
        let hash = hash_prompt(prompt);
        let now = Instant::now();

        let mut entry = self.recent.entry(client_id.to_string()).or_default();

        // Drop anything outside the window before counting.
        while let Some((_, t)) = entry.front() {
            if now.duration_since(*t) > self.window {
                entry.pop_front();
            } else {
                break;
            }
        }

        let repeats = entry.iter().filter(|(h, _)| *h == hash).count();
        entry.push_back((hash, now));
        if entry.len() > MAX_HISTORY_PER_CLIENT {
            entry.pop_front();
        }

        let is_loop = repeats + 1 >= self.threshold;
        if is_loop {
            warn!(
                client_id,
                repeats = repeats + 1,
                window_secs = self.window.as_secs(),
                "LoopGuard: same prompt repeated rapidly — likely a stuck agent"
            );
        }
        is_loop
    }
}

impl Default for LoopGuard {
    fn default() -> Self { Self::new() }
}

fn hash_prompt(prompt: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// SPEND GUARD
// ============================================================================

/// Default: $50 (5,000 cents) per client per rolling hour. Override with
/// MAX_SPEND_CENTS_PER_CLIENT / SPEND_GUARD_WINDOW_SECS env vars — see main.rs.
const DEFAULT_MAX_CENTS_PER_WINDOW: f64 = 5_000.0;
const DEFAULT_SPEND_WINDOW: Duration = Duration::from_secs(3_600);

pub struct SpendGuard {
    spent: DashMap<String, (f64, Instant)>,
    max_cents_per_window: f64,
    window: Duration,
}

impl SpendGuard {
    pub fn new() -> Self {
        Self::with_config(DEFAULT_MAX_CENTS_PER_WINDOW, DEFAULT_SPEND_WINDOW)
    }

    pub fn with_config(max_cents_per_window: f64, window: Duration) -> Self {
        Self { spent: DashMap::new(), max_cents_per_window, window }
    }

    /// Returns `true` if this client is still under their cap and the
    /// request may proceed. Does NOT record anything — call `record_spend`
    /// after a call actually completes and its real cost is known.
    pub fn check(&self, client_id: &str) -> bool {
        match self.spent.get(client_id) {
            None => true,
            Some(entry) => {
                let (spent, window_start) = *entry;
                if Instant::now().duration_since(window_start) > self.window {
                    true // window has rolled over — allow, will reset on next record_spend
                } else {
                    spent < self.max_cents_per_window
                }
            }
        }
    }

    /// Adds `cost_cents` to this client's running total, resetting the
    /// window if it has expired.
    pub fn record_spend(&self, client_id: &str, cost_cents: f64) {
        let now = Instant::now();
        let mut entry = self.spent.entry(client_id.to_string()).or_insert((0.0, now));

        if now.duration_since(entry.1) > self.window {
            entry.0 = 0.0;
            entry.1 = now;
        }
        entry.0 += cost_cents;

        if entry.0 >= self.max_cents_per_window {
            warn!(
                client_id,
                spent_cents = entry.0,
                cap_cents = self.max_cents_per_window,
                "SpendGuard: client has hit its spend cap for this window"
            );
        }
    }
}

impl Default for SpendGuard {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_prompts_trip_after_threshold() {
        let g = LoopGuard::with_config(3, Duration::from_secs(60));
        assert!(!g.check_and_record("client-a", "same prompt"));
        assert!(!g.check_and_record("client-a", "same prompt"));
        assert!(g.check_and_record("client-a", "same prompt")); // 3rd occurrence trips it
    }

    #[test]
    fn different_prompts_do_not_trip() {
        let g = LoopGuard::with_config(3, Duration::from_secs(60));
        assert!(!g.check_and_record("client-b", "prompt one"));
        assert!(!g.check_and_record("client-b", "prompt two"));
        assert!(!g.check_and_record("client-b", "prompt three"));
    }

    #[test]
    fn loops_are_isolated_per_client() {
        let g = LoopGuard::with_config(2, Duration::from_secs(60));
        assert!(!g.check_and_record("client-c", "hello"));
        // A different client repeating the same text shouldn't inherit client-c's count.
        assert!(!g.check_and_record("client-d", "hello"));
    }

    #[test]
    fn spend_guard_blocks_after_cap() {
        let g = SpendGuard::with_config(100.0, Duration::from_secs(3600));
        assert!(g.check("client-e"));
        g.record_spend("client-e", 60.0);
        assert!(g.check("client-e"));
        g.record_spend("client-e", 50.0); // total 110 >= 100
        assert!(!g.check("client-e"));
    }
}
