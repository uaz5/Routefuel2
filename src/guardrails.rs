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
//
// FIX (this revision): SpendGuard previously exposed only check() (a plain
// read) + record_spend() (called after the real cost was known). That's a
// check-then-act race — under concurrent requests from the same client,
// many could all pass check() before any of them called record_spend(),
// letting total spend blow well past the cap. It also meant shadow-mode
// calls (main.rs::maybe_fire_shadow_request) never called check() at all,
// only record_spend() after the fact, so a client already over cap could
// keep triggering fully-billed shadow calls indefinitely.
//
// try_reserve()/reconcile()/release() replace that pattern with an atomic
// reserve-then-adjust: the estimated cost is reserved against the cap in
// the same operation that checks it, then reconciled (or released) once
// the real cost — or the fact that no call happened at all — is known.
// check()/record_spend() are left in place for compatibility but should be
// treated as deprecated; new call sites should use the reserve pattern.
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
    ///
    /// DEPRECATED: this is a plain check with no reservation, so concurrent
    /// callers can all pass it before any of them records real spend. Use
    /// `try_reserve` for new call sites — it checks and reserves in one
    /// atomic step.
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
    ///
    /// DEPRECATED: paired with `check()`. Use `try_reserve`/`reconcile`/
    /// `release` for new call sites.
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

    /// Atomically checks AND reserves `estimated_cost_cents` against the
    /// client's cap in one operation — closes the race where many
    /// concurrent requests all pass a plain check() before any of them
    /// finishes and records its real cost. Returns false (reserving
    /// nothing) if the reservation would exceed the cap.
    pub fn try_reserve(&self, client_id: &str, estimated_cost_cents: f64) -> bool {
        let now = Instant::now();
        let mut entry = self.spent.entry(client_id.to_string()).or_insert((0.0, now));

        if now.duration_since(entry.1) > self.window {
            entry.0 = 0.0;
            entry.1 = now;
        }

        if entry.0 + estimated_cost_cents > self.max_cents_per_window {
            warn!(
                client_id,
                spent_cents = entry.0,
                attempted_cents = estimated_cost_cents,
                cap_cents = self.max_cents_per_window,
                "SpendGuard: reservation would exceed cap — rejecting before any provider call"
            );
            return false;
        }

        entry.0 += estimated_cost_cents;
        true
    }

    /// Adjusts a prior reservation to the real cost once known (a call is
    /// commonly a bit more or less than the pre-call estimate).
    pub fn reconcile(&self, client_id: &str, estimated_cost_cents: f64, actual_cost_cents: f64) {
        let delta = actual_cost_cents - estimated_cost_cents;
        if delta == 0.0 { return; }
        let mut entry = self.spent.entry(client_id.to_string()).or_insert((0.0, Instant::now()));
        entry.0 = (entry.0 + delta).max(0.0);
    }

    /// Releases a reservation entirely — for when the call never happened
    /// (rejected/errored before any provider was billed).
    pub fn release(&self, client_id: &str, estimated_cost_cents: f64) {
        self.reconcile(client_id, estimated_cost_cents, 0.0);
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

    #[test]
    fn try_reserve_blocks_once_cap_would_be_exceeded() {
        let g = SpendGuard::with_config(100.0, Duration::from_secs(3600));
        assert!(g.try_reserve("client-f", 60.0));
        assert!(g.try_reserve("client-f", 30.0)); // running total 90, still under 100
        assert!(!g.try_reserve("client-f", 20.0)); // would push to 110 — rejected, nothing added
    }

    #[test]
    fn concurrent_reservations_cannot_both_pass_a_stale_check() {
        // Simulates the race check()/record_spend() was vulnerable to:
        // two reservations that individually fit under the cap, but not
        // both at once, should not both succeed.
        let g = SpendGuard::with_config(100.0, Duration::from_secs(3600));
        assert!(g.try_reserve("client-g", 70.0));
        assert!(!g.try_reserve("client-g", 70.0)); // second reservation correctly rejected
    }

    #[test]
    fn release_frees_a_reservation() {
        let g = SpendGuard::with_config(100.0, Duration::from_secs(3600));
        assert!(g.try_reserve("client-h", 90.0));
        assert!(!g.try_reserve("client-h", 20.0)); // would exceed cap
        g.release("client-h", 90.0); // e.g. the call failed, nothing was billed
        assert!(g.try_reserve("client-h", 20.0)); // now fits
    }

    #[test]
    fn reconcile_adjusts_reservation_to_real_cost() {
        let g = SpendGuard::with_config(100.0, Duration::from_secs(3600));
        assert!(g.try_reserve("client-i", 50.0)); // estimate
        g.reconcile("client-i", 50.0, 30.0); // actual cost was lower
        assert!(g.try_reserve("client-i", 65.0)); // 30 + 65 = 95, fits under 100
    }
}
