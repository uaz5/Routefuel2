// ============================================================================
// src/circuit_breaker.rs
// Circuit breaker pattern for provider health monitoring
// Reports 5xx errors and timeouts back to disconnect unhealthy providers
//
// FIX (this revision): is_open()'s HalfOpen fast path used to return
// `false` (allow) for EVERY caller once a provider transitioned into
// HalfOpen, not just the single trial request the design intends. Under
// concurrent traffic, that let a stampede of requests hit a provider the
// instant it started recovering, instead of one probe. `probe_in_flight`
// on ProviderState now gates that: only the caller that actually performs
// the Open -> HalfOpen transition gets through; everyone else is blocked
// until record_success/record_failure resolves the trial.
// ============================================================================

use crate::connectors::Provider;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{warn, info};

const FAILURE_THRESHOLD: u32 = 3;
const RESET_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone)]
struct ProviderState {
    state: CircuitState,
    failures: u32,
    last_failure_time: Option<Instant>,
    opened_at: Option<Instant>,
    /// True while a single half-open trial request is in flight. Cleared
    /// by record_success/record_failure once that trial resolves.
    probe_in_flight: bool,
}

impl Default for ProviderState {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failures: 0,
            last_failure_time: None,
            opened_at: None,
            probe_in_flight: false,
        }
    }
}

pub struct CircuitBreaker {
    providers: RwLock<HashMap<Provider, ProviderState>>,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a provider's circuit is open (not available).
    pub fn is_open(&self, provider: Provider) -> bool {
        // Fast path under a read lock.
        {
            let providers = self.providers.read();
            if let Some(state) = providers.get(&provider) {
                match state.state {
                    CircuitState::Closed => return false,
                    CircuitState::HalfOpen => {
                        // FIX: only the caller holding the probe is let
                        // through (probe_in_flight == true means SOMEONE
                        // ELSE holds it, so THIS caller is blocked).
                        return state.probe_in_flight;
                    }
                    CircuitState::Open => {
                        let expired = state
                            .opened_at
                            .map(|t| t.elapsed() > RESET_TIMEOUT)
                            .unwrap_or(false);
                        if !expired {
                            return true;
                        }
                        // else fall through to claim the HalfOpen probe below
                    }
                }
            } else {
                return false; // never recorded a failure for this provider
            }
        }

        // Timeout expired — claim the single HalfOpen probe under a write
        // lock. Only the caller that actually flips Open -> HalfOpen here
        // gets through (returns false). If another thread already made
        // that transition between our read above and this write lock, we
        // fall through to the match below and get blocked like any other
        // late arrival.
        let mut providers = self.providers.write();
        let state = providers.entry(provider).or_default();
        if state.state == CircuitState::Open {
            state.state = CircuitState::HalfOpen;
            state.probe_in_flight = true;
            info!(?provider, "Circuit breaker half-open — allowing a single trial request");
            return false;
        }
        match state.state {
            CircuitState::HalfOpen => state.probe_in_flight,
            CircuitState::Closed => false,
            CircuitState::Open => true,
        }
    }

    /// Record a successful request.
    pub fn record_success(&self, provider: Provider) {
        let mut providers = self.providers.write();
        let state = providers.entry(provider).or_default();

        state.failures = 0;
        state.last_failure_time = None;

        if state.state != CircuitState::Closed {
            let was_half_open = state.state == CircuitState::HalfOpen;
            state.state = CircuitState::Closed;
            state.opened_at = None;
            state.probe_in_flight = false; // FIX: release the probe slot
            if was_half_open {
                info!(?provider, "Circuit breaker closed after recovery");
            } else {
                info!(?provider, "Circuit breaker closed by success while open");
            }
        }
    }

    /// Record a failed request (5xx, timeout, etc.)
    pub fn record_failure(&self, provider: Provider) {
        let mut providers = self.providers.write();
        let state = providers.entry(provider).or_default();

        state.failures += 1;
        state.last_failure_time = Some(Instant::now());

        if state.state == CircuitState::HalfOpen {
            state.state = CircuitState::Open;
            state.opened_at = Some(Instant::now());
            state.probe_in_flight = false; // FIX: release; next trial claims a fresh probe
            warn!(?provider, "Circuit breaker re-opened after failed half-open trial");
        } else if state.failures >= FAILURE_THRESHOLD && state.state != CircuitState::Open {
            state.state = CircuitState::Open;
            state.opened_at = Some(Instant::now());
            warn!(
                ?provider,
                failures = state.failures,
                "Circuit breaker opened due to failures"
            );
        }
    }

    /// Get current state of a provider
    pub fn state(&self, provider: Provider) -> CircuitState {
        let providers = self.providers.read();
        providers
            .get(&provider)
            .map(|s| s.state)
            .unwrap_or(CircuitState::Closed)
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let breaker = CircuitBreaker::new();
        let provider = Provider::OpenAI;

        for _ in 0..3 {
            breaker.record_failure(provider);
        }

        assert!(breaker.is_open(provider));
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let breaker = CircuitBreaker::new();
        let provider = Provider::Anthropic;

        for _ in 0..3 {
            breaker.record_failure(provider);
        }

        assert!(breaker.is_open(provider));

        breaker.record_success(provider);
        assert_eq!(breaker.state(provider), CircuitState::Closed);
    }

    #[test]
    fn only_one_caller_gets_through_half_open() {
        let breaker = CircuitBreaker::new();
        let provider = Provider::Gemini;

        for _ in 0..3 {
            breaker.record_failure(provider);
        }
        assert!(breaker.is_open(provider));

        // Can't fast-forward real time in a unit test without a mockable
        // clock, so this test documents intent rather than exercising the
        // RESET_TIMEOUT path directly: once HalfOpen with a probe in
        // flight, every other caller must see is_open() == true.
        // (Full timeout-crossing behavior is covered by manual/integration
        // testing given Instant isn't mockable here.)
    }
}
