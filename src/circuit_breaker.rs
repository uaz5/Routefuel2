// ============================================================================
// src/circuit_breaker.rs
// Circuit breaker pattern for provider health monitoring
// Reports 5xx errors and timeouts back to disconnect unhealthy providers
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
}

impl Default for ProviderState {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failures: 0,
            last_failure_time: None,
            opened_at: None,
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

    /// Check if a provider's circuit is open (not available)
    pub fn is_open(&self, provider: Provider) -> bool {
        let providers = self.providers.read();
        let state = providers.get(&provider).cloned().unwrap_or_default();

        match state.state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                if let Some(opened_at) = state.opened_at {
                    if opened_at.elapsed() > RESET_TIMEOUT {
                        // Timeout expired, try half-open
                        return false;
                    }
                }
                true
            }
            CircuitState::HalfOpen => false,
        }
    }

    /// Record a successful request
    pub fn record_success(&self, provider: Provider) {
        let mut providers = self.providers.write();
        let state = providers.entry(provider).or_default();

        state.failures = 0;
        state.last_failure_time = None;

        if state.state == CircuitState::HalfOpen {
            state.state = CircuitState::Closed;
            state.opened_at = None;
            info!(?provider, "Circuit breaker closed after recovery");
        }
    }

    /// Record a failed request (5xx, timeout, etc.)
    pub fn record_failure(&self, provider: Provider) {
        let mut providers = self.providers.write();
        let state = providers.entry(provider).or_default();

        state.failures += 1;
        state.last_failure_time = Some(Instant::now());

        if state.failures >= FAILURE_THRESHOLD && state.state != CircuitState::Open {
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
}
