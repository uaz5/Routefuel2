// ============================================================================
// src/concurrency.rs — RouteFuel v0.8
//
// Bounds how many provider calls can be in flight at once, across BOTH the
// non-streaming path (main.rs's handle_non_streaming) and the streaming path
// (streaming.rs's stream_handler) — that's why this lives at the AppState
// level rather than inside ConnectorManager, which the streaming path never
// touches (it talks to providers directly for full control over SSE parsing).
//
// Without this, a traffic spike means RouteFuel opens as many simultaneous
// HTTP connections to providers as it has incoming requests — which is
// exactly how you get rate-limited or IP-blocked by a provider during a
// burst, on top of however many client keys are hammering it at once. A
// semaphore-backed permit makes request #(N+1) wait for a slot instead of
// firing immediately, turning a thundering herd into a queue.
//
// For streaming, the permit is acquired *inside* the SSE generator itself
// (see streaming.rs) so it's held for the full lifetime of the stream, not
// just the initial connect — a long-lived stream correctly counts against
// the pool for as long as it's open, and releases automatically the moment
// the stream ends or the client disconnects (axum drops the generator,
// which drops the permit).
// ============================================================================

use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::debug;

pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
    max: usize,
}

impl ConcurrencyLimiter {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent.max(1))),
            max: max_concurrent.max(1),
        }
    }

    /// Waits until a slot is free, then returns a permit that releases the
    /// slot automatically when dropped. Hold it for exactly as long as the
    /// provider call (or the whole streamed response) is live — no longer,
    /// no shorter.
    pub async fn acquire(&self) -> OwnedSemaphorePermit {
        if self.semaphore.available_permits() == 0 {
            debug!(max = self.max, "Concurrency limit reached — request queued");
        }
        Arc::clone(&self.semaphore)
            .acquire_owned()
            .await
            .expect("semaphore is never explicitly closed, so this cannot fail")
    }

    pub fn available(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn max(&self) -> usize {
        self.max
    }

    /// How many calls are currently in flight — useful for a health/metrics
    /// endpoint.
    pub fn in_flight(&self) -> usize {
        self.max.saturating_sub(self.available())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn permits_are_bounded_and_released() {
        let limiter = ConcurrencyLimiter::new(2);
        assert_eq!(limiter.available(), 2);

        let p1 = limiter.acquire().await;
        let p2 = limiter.acquire().await;
        assert_eq!(limiter.available(), 0);
        assert_eq!(limiter.in_flight(), 2);

        drop(p1);
        assert_eq!(limiter.available(), 1);

        drop(p2);
        assert_eq!(limiter.available(), 2);
    }

    #[tokio::test]
    async fn third_acquire_waits_for_a_slot() {
        let limiter = Arc::new(ConcurrencyLimiter::new(1));
        let p1 = limiter.acquire().await;

        let l2 = Arc::clone(&limiter);
        let waiter = tokio::spawn(async move {
            let _p2 = l2.acquire().await; // should block until p1 drops
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "second acquire should still be waiting");

        drop(p1);
        tokio::time::timeout(Duration::from_millis(200), waiter)
            .await
            .expect("waiter should complete shortly after the permit is released")
            .unwrap();
    }
}
