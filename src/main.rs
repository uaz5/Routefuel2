// ============================================================================
// src/main.rs
// RouterFuel production server — pure BYOK, zero gateway-side provider spend
// ============================================================================

mod admin;
mod auth;
mod circuit_breaker;
mod client_registry;
mod concurrency;
mod connectors;
mod cost_tracker;
mod embedder;
mod guardrails;
mod openrouter_catalog;
mod rate_limiter;
mod route_engine;
mod semantic_cache;
mod streaming;
mod telemetry;
mod tokens;
mod vision;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tower_http::{
    cors::CorsLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::admin::AdminState;
use crate::auth::{api_key_middleware, ApiKeyStore, ClientProviderKeys};
use crate::circuit_breaker::CircuitBreaker;
use crate::concurrency::ConcurrencyLimiter;
use crate::connectors::{
    ChatCompletionRequest, ChatCompletionResponse, ConnectorManager, Provider,
};
use crate::cost_tracker::{CostTracker, ShadowComparison};
use crate::embedder::LocalEmbedder;
use crate::guardrails::{LoopGuard, SpendGuard};
use crate::rate_limiter::{RateLimiter, TierConfig};
use crate::route_engine::{MeetingTask, RouteEngine, RoutingPriority};
use crate::semantic_cache::SemanticCache;
use crate::telemetry::{TelemetryData, TelemetryRecorder};
use crate::tokens::TokenCostBreakdown;

// ============================================================================
// APPLICATION STATE
// ============================================================================

#[derive(Clone)]
struct AppState {
    route_engine: Arc<RouteEngine>,
    connector_manager: Arc<ConnectorManager>,
    cost_tracker: Arc<CostTracker>,
    circuit_breaker: Arc<CircuitBreaker>,
    semantic_cache: Arc<SemanticCache>,
    rate_limiter: Arc<RateLimiter>,
    loop_guard: Arc<LoopGuard>,
    spend_guard: Arc<SpendGuard>,
    telemetry: Arc<TelemetryRecorder>,
    concurrency_limiter: Arc<ConcurrencyLimiter>,
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(serde::Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

enum ApiError {
    BadRequest(String),
    RateLimited,
    LoopDetected,
    SpendCapExceeded,
    CircuitOpen,
    ProviderError(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, error_type) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg, "invalid_request_error"),
            ApiError::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded".to_string(),
                "rate_limit_error",
            ),
            ApiError::LoopDetected => (
                StatusCode::TOO_MANY_REQUESTS,
                "This looks like an agent stuck in a loop: the same prompt has been sent \
                 several times in the last minute. The request was blocked before calling \
                 any provider, so nothing was billed for it. If this is intentional (e.g. \
                 polling), space out retries or vary the prompt."
                    .to_string(),
                "loop_detected_error",
            ),
            ApiError::SpendCapExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                "This client has hit its spend cap for the current window. This usually means \
                 an agent is stuck retrying or looping and about to run up a large bill — the \
                 request was blocked before any provider was called. Contact the gateway \
                 operator to raise the cap if this is expected traffic."
                    .to_string(),
                "spend_cap_error",
            ),
            ApiError::CircuitOpen => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Service temporarily unavailable".to_string(),
                "server_error",
            ),
            ApiError::ProviderError(msg) => (StatusCode::BAD_GATEWAY, msg, "provider_error"),
            ApiError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                msg,
                "internal_error",
            ),
        };

        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
            },
        });

        (status, body).into_response()
    }
}

// ============================================================================
// BYOK RESOLUTION
//
// RouterFuel never holds a paid provider key of its own. Every call must be
// billed to the client's own account:
//   1. Client supplied a key for the exact provider RouterFuel selected → use it.
//   2. Client didn't, but supplied an OpenRouter key → re-route the SAME
//      model through OpenRouter (model id becomes "<vendor>/<model>"), still
//      billed entirely to the client.
//   3. Neither → reject with 400 before any connector is touched. There is
//      no fallback to a gateway-owned key anywhere in this codebase.
// ============================================================================

struct ByokRoute {
    provider_to_call: Provider,
    model_id_to_send: String,
    api_key: String,
    used_openrouter_fallback: bool,
}

fn resolve_byok_route(
    selected_provider: Provider,
    requested_model: &str,
    keys: &ClientProviderKeys,
) -> Result<ByokRoute, ApiError> {
    if let Some(key) = keys.for_provider(selected_provider) {
        return Ok(ByokRoute {
            provider_to_call: selected_provider,
            model_id_to_send: requested_model.to_string(),
            api_key: key.to_string(),
            used_openrouter_fallback: false,
        });
    }

    if let Some(or_key) = keys.openrouter.as_deref() {
        let model_id_to_send = if selected_provider == Provider::OpenRouter {
            requested_model.to_string()
        } else {
            format!("{}/{}", selected_provider.openrouter_prefix(), requested_model)
        };
        return Ok(ByokRoute {
            provider_to_call: Provider::OpenRouter,
            model_id_to_send,
            api_key: or_key.to_string(),
            used_openrouter_fallback: true,
        });
    }

    Err(ApiError::BadRequest(format!(
        "No API key supplied for provider '{selected_provider}'. RouterFuel is fully bring-your-own-key: \
         it never bills its own account for provider calls. Supply your key via the \
         X-{selected_provider}-Api-Key header, or supply an X-Openrouter-Api-Key to route this model \
         through OpenRouter instead."
    )))
}

// ============================================================================
// CHAT COMPLETIONS — TOP-LEVEL DISPATCH (streaming vs. non-streaming)
// ============================================================================

async fn chat_completions_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.stream.unwrap_or(false) {
        return handle_streaming(headers, state, request).await;
    }
    handle_non_streaming(headers, state, request).await.into_response()
}

/// The streaming twin of `handle_non_streaming` below: same guardrails and
/// BYOK resolution, but handed off to `streaming::stream_handler` for the
/// actual SSE proxy instead of returning a single JSON body. Semantic cache
/// is intentionally not consulted here — caching a streamed response would
/// mean buffering the whole thing anyway, which defeats the point of
/// streaming; a cache hit for the same prompt still short-circuits the next
/// *non*-streaming call.
async fn handle_streaming(headers: HeaderMap, state: AppState, request: ChatCompletionRequest) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let client_id = headers
        .get("x-routerfuel-client-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let rl_key = client_id.clone().unwrap_or_else(|| "anonymous".to_string());

    if state.rate_limiter.check(&rl_key).is_err() {
        warn!(client_id = %rl_key, "Rate limit exceeded (streaming)");
        return ApiError::RateLimited.into_response();
    }
    if !state.spend_guard.check(&rl_key) {
        return ApiError::SpendCapExceeded.into_response();
    }

    let prompt_text = request
        .messages
        .iter()
        .filter(|m| m.role == "user")
        .last()
        .map(|m| m.content.as_text())
        .unwrap_or_default();

    if !prompt_text.is_empty() && state.loop_guard.check_and_record(&rl_key, &prompt_text) {
        return ApiError::LoopDetected.into_response();
    }

    let (selected_provider, routing_model_id) = match resolve_model(&state, &request, 0) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };

    let provider_keys = ClientProviderKeys::from_headers(&headers);
    let byok = match resolve_byok_route(selected_provider, &routing_model_id, &provider_keys) {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };

    let mut effective_request = request.clone();
    effective_request.model = byok.model_id_to_send.clone();

    info!(
        request_id = %request_id,
        provider = ?byok.provider_to_call,
        model = %effective_request.model,
        "Starting streaming request"
    );

    streaming::stream_handler(
        request_id,
        byok.provider_to_call,
        effective_request.model.clone(),
        byok.api_key.clone(),
        effective_request,
        client_id,
        true, // is_byok — RouterFuel is BYOK-only, see resolve_byok_route
        Arc::clone(&state.route_engine),
        Arc::clone(&state.cost_tracker),
        reqwest::Client::new(),
        Arc::clone(&state.concurrency_limiter),
    )
    .await
    .into_response()
}

/// Resolves what to route to, supporting three forms in the `model` field:
///   - a concrete model id, e.g. "claude-opus-4-8"          → routed as-is
///   - "auto"                                                → best model by
///     balanced cost/latency/quality score, given the real input token count
///   - "task:<name>", e.g. "task:summarise"                  → the model
///     RouterFuel has pre-selected as best-fit for that task (see
///     route_engine::MeetingTask and RouteEngine::select_for_task)
fn resolve_model(
    state: &AppState,
    request: &ChatCompletionRequest,
    input_tokens: u32,
) -> Result<(Provider, String), ApiError> {
    // Previously nothing checked whether an image-carrying request landed
    // on a vision-capable model — vision.rs's select_vision_model existed
    // but was never called from the request path. See src/vision.rs.
    let has_image = request.messages.iter().any(|m| m.content.has_image());

    if request.model == "auto" {
        let decision = if has_image {
            crate::vision::select_vision_model(&state.route_engine, input_tokens, RoutingPriority::Balanced)
                .map_err(|e| ApiError::ProviderError(format!("No vision-capable provider available: {e}")))?
        } else {
            state
                .route_engine
                .select(input_tokens, request.max_tokens.unwrap_or(1024), RoutingPriority::Balanced)
                .map_err(|e| ApiError::ProviderError(format!("No available providers: {e}")))?
        };
        return Ok((decision.model.provider, decision.model.api_id));
    }

    if let Some(task_str) = request.model.strip_prefix("task:") {
        let task: MeetingTask = task_str
            .parse()
            .map_err(|e: anyhow::Error| ApiError::BadRequest(e.to_string()))?;
        let decision = state
            .route_engine
            .select_for_task(task, input_tokens)
            .map_err(|e| ApiError::ProviderError(format!("Task routing failed: {e}")))?;

        if has_image && !decision.model.supports_vision {
            let fallback =
                crate::vision::select_vision_model(&state.route_engine, input_tokens, RoutingPriority::Balanced)
                    .map_err(|e| ApiError::ProviderError(format!("No vision-capable provider available: {e}")))?;
            return Ok((fallback.model.provider, fallback.model.api_id));
        }
        return Ok((decision.model.provider, decision.model.api_id));
    }

    let provider = state
        .route_engine
        .select_provider(&request.model)
        .map_err(|e| {
            error!("Routing failed: {}", e);
            ApiError::ProviderError("No available providers".to_string())
        })?;

    // Client pinned a concrete model id — if it can't take images and the
    // request has one, fail clearly instead of silently sending an image to
    // a text-only model (which just gets ignored or errors deep inside the
    // provider call with a much less useful message).
    if has_image {
        if let Ok(model) = state.route_engine.find(&request.model) {
            if !model.supports_vision {
                return Err(ApiError::BadRequest(format!(
                    "Model '{}' does not support image input. Use \"model\": \"auto\" to let \
                     RouterFuel pick a vision-capable model, or choose one directly (see GET /v1/models \
                     for supports_vision).",
                    request.model
                )));
            }
        }
    }

    Ok((provider, request.model.clone()))
}

// ============================================================================
// CHAT COMPLETIONS HANDLER (non-streaming)
// ============================================================================

#[instrument(skip(headers, state, request), fields(request_id))]
async fn handle_non_streaming(
    headers: HeaderMap,
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let request_id = Uuid::new_v4().to_string();
    tracing::Span::current().record("request_id", &request_id);

    let start = Instant::now();

    // Extract authenticated client ID injected by auth middleware
    let client_id = headers
        .get("x-routerfuel-client-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    info!(
        model = %request.model,
        client_id = ?client_id,
        message_count = request.messages.len(),
        "Received chat completion request"
    );

    // ========================================================================
    // STEP -1: RATE LIMIT, LOOP GUARD, SPEND CAP — all cheap, all before
    // anything that costs money or calls a provider.
    // ========================================================================
    let rl_key = client_id.clone().unwrap_or_else(|| "anonymous".to_string());

    if state.rate_limiter.check(&rl_key).is_err() {
        warn!(client_id = %rl_key, "Rate limit exceeded");
        return Err(ApiError::RateLimited);
    }

    if !state.spend_guard.check(&rl_key) {
        return Err(ApiError::SpendCapExceeded);
    }

    let prompt_text = request
        .messages
        .iter()
        .filter(|m| m.role == "user")
        .last()
        .map(|m| m.content.as_text())
        .unwrap_or_default();

    if !prompt_text.is_empty() && state.loop_guard.check_and_record(&rl_key, &prompt_text) {
        return Err(ApiError::LoopDetected);
    }

    // ========================================================================
    // STEP 0: CHECK SEMANTIC CACHE (local ONNX embedding, ~1-5ms, $0 — no
    // provider is called and nothing is billed to anyone on a cache hit)
    // ========================================================================
    if !prompt_text.is_empty() {
        if let Some(hit) = state.semantic_cache.lookup(&prompt_text).await {
            info!(
                request_id = %request_id,
                similarity = hit.similarity,
                model = %hit.model_used,
                "Semantic cache hit! Serving instantly (bypassing LLM)"
            );

            let cached_response: ChatCompletionResponse = serde_json::from_str(&hit.cached_response)
                .map_err(|e| ApiError::InternalError(format!("Failed to parse cache: {}", e)))?;

            return Ok(Json(cached_response));
        }
    }

    // ========================================================================
    // STEP 1: COUNT INPUT TOKENS PRECISELY
    // ========================================================================

    let input_tokens = tokens::count_request_tokens(&request.messages, &request.model)
        .map_err(|e| {
            error!("Token counting failed: {}", e);
            ApiError::InternalError(format!("Token counting failed: {}", e))
        })?;

    let estimated_output = tokens::estimate_output_tokens(request.max_tokens, &request.model);

    debug!(
        input_tokens = input_tokens,
        estimated_output_tokens = estimated_output,
        "Counted request tokens"
    );

    // ========================================================================
    // STEP 2: ROUTING DECISION (<10ms target)
    // ========================================================================

    let routing_start = Instant::now();

    let (selected_provider, routing_model_id) = resolve_model(&state, &request, input_tokens)?;

    let routing_decision_ms = routing_start.elapsed().as_millis() as u64;

    debug!(
        selected_provider = ?selected_provider,
        routing_model_id = %routing_model_id,
        routing_decision_ms = routing_decision_ms,
        "Routing decision completed"
    );

    if routing_decision_ms > 10 {
        error!(
            "Routing decision exceeded 10ms target: {}ms",
            routing_decision_ms
        );
    }

    // ========================================================================
    // STEP 3: RESOLVE BYOK ROUTE (mandatory — see resolve_byok_route above)
    // ========================================================================

    let provider_keys = ClientProviderKeys::from_headers(&headers);
    let byok = resolve_byok_route(selected_provider, &routing_model_id, &provider_keys)?;

    if byok.used_openrouter_fallback {
        info!(
            request_id = %request_id,
            original_model = %request.model,
            openrouter_model = %byok.model_id_to_send,
            "No direct key for selected provider — routing via client's OpenRouter key"
        );
    }

    let mut effective_request = request.clone();
    effective_request.model = byok.model_id_to_send.clone();

    // Bounded concurrency (src/concurrency.rs) — waits here if RouterFuel is
    // already at MAX_CONCURRENT_PROVIDER_CALLS in-flight requests, instead
    // of firing an unbounded number of simultaneous provider connections.
    let _permit = state.concurrency_limiter.acquire().await;

    let connector_result = state
        .connector_manager
        .call(byok.provider_to_call, &effective_request, &byok.api_key)
        .await
        .map_err(|e| {
            error!("Connector error: {}", e);

            state.cost_tracker.record_error(
                request_id.clone(),
                byok.provider_to_call,
                routing_model_id.clone(),
                e.to_string(),
                start.elapsed().as_millis() as u64,
                routing_decision_ms,
                client_id.clone(),
            );

            match e {
                connectors::ConnectorError::CircuitOpen => ApiError::CircuitOpen,
                connectors::ConnectorError::RateLimited => ApiError::RateLimited,
                connectors::ConnectorError::Unauthorized => ApiError::BadRequest(
                    "The BYOK API key supplied for this provider was rejected. Check the key and try again.".to_string(),
                ),
                connectors::ConnectorError::ServerError { status } => {
                    ApiError::ProviderError(format!("Provider returned error status {}", status))
                }
                _ => ApiError::ProviderError(e.to_string()),
            }
        })?;

    let response = connector_result.response.clone();
    let latency_ms = connector_result.latency_ms;
    let output_tokens = connector_result.output_tokens;

    // ========================================================================
    // STORE IN SEMANTIC CACHE (Background non-blocking task)
    // ========================================================================
    if !prompt_text.is_empty() {
        if let Ok(response_json) = serde_json::to_string(&response) {
            state.semantic_cache.store(
                prompt_text,
                response_json,
                response.model.clone(),
            );
        }
    }

    // ========================================================================
    // STEP 4: VERIFY OUTPUT TOKENS
    // ========================================================================

    let response_text = response
        .choices
        .first()
        .map(|c| c.message.content.as_text())
        .unwrap_or_default();

    if let Ok((counted, matches)) = tokens::verify_output_tokens(&response_text, output_tokens) {
        debug!(
            reported = output_tokens,
            counted = counted,
            matches = matches,
            "Verified output tokens"
        );
    }

    // ========================================================================
    // STEP 5: CALCULATE COST WITH PRECISE TOKENS
    // (RouterFuel never pays this — it's billed to the client's own BYOK key.
    // We still track it for the client's own audit/savings dashboard.)
    // ========================================================================

    let (cost_per_1m_input, cost_per_1m_output) = state
        .route_engine
        .get_pricing(&routing_model_id)
        .map_err(|e| {
            error!("Pricing lookup failed: {}", e);
            ApiError::InternalError("Pricing lookup failed".to_string())
        })?;

    let token_cost =
        TokenCostBreakdown::new(input_tokens, output_tokens, cost_per_1m_input, cost_per_1m_output);

    let baseline_cost =
        TokenCostBreakdown::new(input_tokens, output_tokens, 500.0, 3000.0); // vs. a flagship model, e.g. GPT-5.5 / Claude Opus tier

    let cost_saved = baseline_cost.total_cost_cents - token_cost.total_cost_cents;
    let savings_pct = if baseline_cost.total_cost_cents > 0.0 {
        (cost_saved / baseline_cost.total_cost_cents) * 100.0
    } else {
        0.0
    };

    debug!(
        cost_cents = token_cost.total_cost_cents,
        baseline_cents = baseline_cost.total_cost_cents,
        cost_saved_cents = cost_saved,
        savings_pct = savings_pct,
        used_openrouter_fallback = byok.used_openrouter_fallback,
        "Calculated costs (billed to client's own BYOK key, not RouterFuel)"
    );

    state.spend_guard.record_spend(&rl_key, token_cost.total_cost_cents);

    // ========================================================================
    // TELEMETRY (JSONL side-channel — see src/telemetry.rs; separate from
    // the Postgres request_logs audit trail, useful for local ROI reports
    // without a DB round trip). Fire-and-forget: never adds latency here.
    // ========================================================================
    {
        let mut telemetry_data = TelemetryData::new(
            request_id.clone(),
            response.model.clone(),
            token_cost.total_cost_cents,
            baseline_cost.total_cost_cents,
            rl_key.clone(),
        );
        telemetry_data.latency_ms = latency_ms;
        telemetry_data.priority = "balanced".to_string();
        telemetry_data.provider = byok.provider_to_call.to_string();
        telemetry_data.success = true;
        telemetry_data.tokens_used = input_tokens + output_tokens;

        let telemetry = Arc::clone(&state.telemetry);
        tokio::spawn(async move {
            if let Err(e) = telemetry.record(telemetry_data).await {
                debug!("Telemetry record failed (non-fatal): {e}");
            }
        });
    }

    // ========================================================================
    // SHADOW MODE — if the client set `shadow_model`, fire an identical
    // request at it in the background purely for comparison. Never blocks
    // or affects the response already computed above; see
    // maybe_fire_shadow_request and migrations/006_shadow_comparisons.sql.
    // Costs the client a second real bill — that's inherent to what shadow
    // mode is (you're paying to find out what the alternative would have
    // cost), not a bug; ENABLE_SHADOW_MODE lets an operator kill it globally.
    // ========================================================================
    if shadow_mode_enabled() {
        if let Some(shadow_model_id) = request.shadow_model.clone() {
            let output_chars = response
                .choices
                .first()
                .map(|c| c.message.content.as_text().len())
                .unwrap_or(0);

            maybe_fire_shadow_request(
                &state,
                request_id.clone(),
                client_id.clone(),
                provider_keys.clone(),
                shadow_model_id,
                &effective_request,
                response.model.clone(),
                byok.provider_to_call,
                token_cost.total_cost_cents,
                latency_ms,
                output_chars,
            );
        }
    }

    // ========================================================================
    // STEP 6: RECORD TO POSTGRES (non-blocking via tokio::spawn)
    // ========================================================================

    state.cost_tracker.record_request(
        request_id.clone(),
        byok.provider_to_call,
        response.model.clone(),
        &token_cost,
        baseline_cost.total_cost_cents,
        latency_ms,
        routing_decision_ms,
        client_id,
        None,
        None,
        true, // is_byok — RouterFuel is BYOK-only; every completed call was billed to a client key
    );

    // ========================================================================
    // STEP 7: RETURN RESPONSE
    // ========================================================================

    let total_latency = start.elapsed().as_millis() as u64;

    info!(
        request_id = %request_id,
        provider = ?byok.provider_to_call,
        latency_ms = total_latency,
        "Request completed successfully"
    );

    Ok(Json(response))
}

// ============================================================================
// SHADOW MODE
//
// A client sets `shadow_model` on a request; RouterFuel fires an identical
// request at that model *in addition to* the normally-routed one, purely
// for comparison — the client only ever sees the primary response, and this
// never blocks it or can make it fail. Useful for answering "would a
// cheaper model have given basically the same answer?" with real traffic
// instead of a synthetic eval.
//
// Costs the client a second real bill — that's inherent to what shadow mode
// is, not a bug. ENABLE_SHADOW_MODE=false disables it globally without a
// code change if that's not something you want clients able to trigger.
// ============================================================================

fn shadow_mode_enabled() -> bool {
    std::env::var("ENABLE_SHADOW_MODE")
        .map(|v| v != "false" && v != "0")
        .unwrap_or(true)
}

#[allow(clippy::too_many_arguments)]
fn maybe_fire_shadow_request(
    state: &AppState,
    request_id: String,
    client_id: Option<String>,
    provider_keys: ClientProviderKeys,
    shadow_model_id: String,
    effective_request: &ChatCompletionRequest,
    primary_model: String,
    primary_provider: Provider,
    primary_cost_cents: f64,
    primary_latency_ms: u64,
    primary_output_chars: usize,
) {
    let route_engine = Arc::clone(&state.route_engine);
    let connector_manager = Arc::clone(&state.connector_manager);
    let cost_tracker = Arc::clone(&state.cost_tracker);
    let spend_guard = Arc::clone(&state.spend_guard);
    let concurrency_limiter = Arc::clone(&state.concurrency_limiter);

    let mut shadow_request = effective_request.clone();
    let spend_key = client_id.clone().unwrap_or_else(|| "anonymous".to_string());

    tokio::spawn(async move {
        let shadow_provider = match route_engine.select_provider(&shadow_model_id) {
            Ok(p) => p,
            Err(e) => {
                debug!(shadow_model = %shadow_model_id, "Shadow mode: unknown model ({e}), skipping");
                cost_tracker.record_shadow_comparison(ShadowComparison {
                    request_id,
                    client_id,
                    primary_model,
                    primary_provider: primary_provider.to_string(),
                    primary_cost_cents,
                    primary_latency_ms,
                    primary_output_chars,
                    shadow_model: shadow_model_id,
                    shadow_provider: "unknown".to_string(),
                    shadow_cost_cents: None,
                    shadow_latency_ms: None,
                    shadow_output_chars: None,
                    shadow_error: Some(format!("unknown shadow model: {e}")),
                });
                return;
            }
        };

        let byok_shadow = match resolve_byok_route(shadow_provider, &shadow_model_id, &provider_keys) {
            Ok(b) => b,
            Err(_) => {
                // Client has no key for the shadow provider (direct or via
                // OpenRouter) — this is expected and common, not an error
                // worth surfacing loudly; just skip.
                debug!(shadow_model = %shadow_model_id, "Shadow mode: no BYOK key for shadow provider, skipping");
                cost_tracker.record_shadow_comparison(ShadowComparison {
                    request_id,
                    client_id,
                    primary_model,
                    primary_provider: primary_provider.to_string(),
                    primary_cost_cents,
                    primary_latency_ms,
                    primary_output_chars,
                    shadow_model: shadow_model_id,
                    shadow_provider: shadow_provider.to_string(),
                    shadow_cost_cents: None,
                    shadow_latency_ms: None,
                    shadow_output_chars: None,
                    shadow_error: Some("no BYOK key available for shadow provider".to_string()),
                });
                return;
            }
        };

        shadow_request.model = byok_shadow.model_id_to_send.clone();
        shadow_request.stream = Some(false);
        shadow_request.shadow_model = None; // don't recurse into another shadow call

        let _permit = concurrency_limiter.acquire().await;

        let call_start = Instant::now();
        let result = connector_manager
            .call(byok_shadow.provider_to_call, &shadow_request, &byok_shadow.api_key)
            .await;
        let shadow_latency_ms = call_start.elapsed().as_millis() as u64;

        match result {
            Ok(connector_result) => {
                let (cost_in, cost_out) = route_engine
                    .get_pricing(&shadow_model_id)
                    .unwrap_or((500.0, 3000.0));
                let shadow_cost = TokenCostBreakdown::new(
                    connector_result.input_tokens,
                    connector_result.output_tokens,
                    cost_in,
                    cost_out,
                );
                let output_chars = connector_result
                    .response
                    .choices
                    .first()
                    .map(|c| c.message.content.as_text().len())
                    .unwrap_or(0);

                // This is a second real, billed call — count it against the
                // client's spend cap same as any other request.
                spend_guard.record_spend(&spend_key, shadow_cost.total_cost_cents);

                debug!(
                    shadow_model = %shadow_model_id,
                    cost_delta_cents = shadow_cost.total_cost_cents - primary_cost_cents,
                    latency_delta_ms = shadow_latency_ms as i64 - primary_latency_ms as i64,
                    "Shadow comparison complete"
                );

                cost_tracker.record_shadow_comparison(ShadowComparison {
                    request_id,
                    client_id,
                    primary_model,
                    primary_provider: primary_provider.to_string(),
                    primary_cost_cents,
                    primary_latency_ms,
                    primary_output_chars,
                    shadow_model: shadow_model_id,
                    shadow_provider: byok_shadow.provider_to_call.to_string(),
                    shadow_cost_cents: Some(shadow_cost.total_cost_cents),
                    shadow_latency_ms: Some(shadow_latency_ms),
                    shadow_output_chars: Some(output_chars),
                    shadow_error: None,
                });
            }
            Err(e) => {
                cost_tracker.record_shadow_comparison(ShadowComparison {
                    request_id,
                    client_id,
                    primary_model,
                    primary_provider: primary_provider.to_string(),
                    primary_cost_cents,
                    primary_latency_ms,
                    primary_output_chars,
                    shadow_model: shadow_model_id,
                    shadow_provider: byok_shadow.provider_to_call.to_string(),
                    shadow_cost_cents: None,
                    shadow_latency_ms: None,
                    shadow_output_chars: None,
                    shadow_error: Some(e.to_string()),
                });
            }
        }
    });
}

// ============================================================================
// HEALTH CHECK & AUDIT
// ============================================================================

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "byok_only": true,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models: Vec<serde_json::Value> = state
        .route_engine
        .list_enabled()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.api_id,
                "display_name": m.display_name,
                "provider": m.provider.to_string(),
                "context_window": m.context_window,
                "supports_vision": m.supports_vision,
                "open_weight": m.open_weight,
                "cost_per_1m_input_cents": m.cost_per_1m_input,
                "cost_per_1m_output_cents": m.cost_per_1m_output,
            })
        })
        .collect();

    Json(json!({ "object": "list", "data": models }))
}

async fn audit_daily_handler(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let date = params
        .get("date")
        .ok_or_else(|| ApiError::BadRequest("date parameter required".to_string()))?;

    let report = state
        .cost_tracker
        .get_daily_report(date)
        .await
        .map_err(|e| {
            error!("Failed to generate report: {}", e);
            ApiError::InternalError("Report generation failed".to_string())
        })?;

    Ok(Json(serde_json::to_value(report).unwrap()))
}

// ============================================================================
// MAIN SERVER
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_target(true)
        .with_level(true)
        .with_thread_ids(true)
        .init();

    info!("Starting RouterFuel v{} (BYOK-only — no gateway provider keys)", env!("CARGO_PKG_VERSION"));

    dotenv::dotenv().ok();

    // NOTE: RouterFuel intentionally reads no OPENAI_API_KEY / ANTHROPIC_API_KEY
    // / etc. here. It never holds a provider key of its own — see
    // connectors.rs and the BYOK resolution logic in this file. The only
    // secrets it needs are its own database and its own client auth store.
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL environment variable not set");

    let routerfuel_keys_raw = std::env::var("ROUTERFUEL_API_KEYS").unwrap_or_default();
    let api_key_store = Arc::new(ApiKeyStore::from_env_string(&routerfuel_keys_raw));

    info!("Configuration loaded from environment");

    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(20)
        .connect(&database_url)
        .await?;

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await?;

    info!("Database migrations completed");

    let circuit_breaker = Arc::new(CircuitBreaker::new());
    let route_engine = Arc::new(RouteEngine::new());
    let cost_tracker = Arc::new(CostTracker::new(pool.clone()));
    let rate_limiter = Arc::new(RateLimiter::new());

    // Per-client rate-limit tiers — from ROUTERFUEL_CLIENT_TIERS and/or the
    // client_tiers table (migrations/003_client_tiers.sql already creates
    // it). Previously main.rs just hardcoded UserTier::Pro for everyone;
    // this is what actually makes per-client tiers real. See
    // src/client_registry.rs.
    let client_tiers_raw = std::env::var("ROUTERFUEL_CLIENT_TIERS").unwrap_or_default();
    client_registry::load_all_tiers(&pool, &rate_limiter, &client_tiers_raw, TierConfig::PRO).await;

    // Runaway-agent protection — see src/guardrails.rs. Both are cheap,
    // in-memory, per-process checks that run before any provider is called.
    let loop_repeat_threshold: usize = std::env::var("LOOP_GUARD_REPEAT_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let loop_window_secs: u64 = std::env::var("LOOP_GUARD_WINDOW_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let loop_guard = Arc::new(guardrails::LoopGuard::with_config(
        loop_repeat_threshold,
        std::time::Duration::from_secs(loop_window_secs),
    ));

    let max_spend_cents_per_client: f64 = std::env::var("MAX_SPEND_CENTS_PER_CLIENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000.0); // $50 default — tune per your client base
    let spend_window_secs: u64 = std::env::var("SPEND_GUARD_WINDOW_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_600);
    let spend_guard = Arc::new(guardrails::SpendGuard::with_config(
        max_spend_cents_per_client,
        std::time::Duration::from_secs(spend_window_secs),
    ));

    info!(
        loop_repeat_threshold,
        loop_window_secs,
        max_spend_cents_per_client,
        spend_window_secs,
        "Runaway-agent guardrails configured"
    );

    // Pull the full public OpenRouter catalog (300+ models) so BYOK clients
    // who only hold an OpenRouter key still get first-class routing across
    // everything OpenRouter hosts, not just the curated direct-integration
    // list in route_engine.rs. Non-fatal on failure — the curated registry
    // works fine without it, and this is a live third-party endpoint.
    let catalog_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    match tokio::time::timeout(
        std::time::Duration::from_secs(12),
        openrouter_catalog::fetch_openrouter_catalog(&catalog_client),
    )
    .await
    {
        Ok(Ok(extra)) => {
            let fetched = extra.len();
            let added = route_engine.extend_registry(extra);
            info!(fetched, added, "Merged OpenRouter catalog into the model registry");
        }
        Ok(Err(e)) => {
            warn!("Could not fetch OpenRouter model catalog at startup ({e}). Continuing with the curated registry only.");
        }
        Err(_) => {
            warn!("OpenRouter model catalog fetch timed out. Continuing with the curated registry only.");
        }
    }

    // Local ONNX embedder for the semantic cache. If the model files aren't
    // present yet (see src/embedder.rs for what to download), the server
    // still starts — the cache just runs in "always miss" mode instead of
    // failing to boot.
    let embedding_model_path = std::env::var("EMBEDDING_MODEL_PATH")
        .unwrap_or_else(|_| "./models/embedding.onnx".to_string());
    let embedding_tokenizer_path = std::env::var("EMBEDDING_TOKENIZER_PATH")
        .unwrap_or_else(|_| "./models/tokenizer.json".to_string());

    let embedder = match LocalEmbedder::load(&embedding_model_path, &embedding_tokenizer_path) {
        Ok(e) => {
            info!("Local ONNX embedding model loaded — semantic cache active");
            Some(Arc::new(e))
        }
        Err(e) => {
            warn!(
                "Could not load local embedding model ({e}). Semantic cache disabled until \
                 {embedding_model_path} and {embedding_tokenizer_path} are present."
            );
            None
        }
    };

    let semantic_cache = Arc::new(SemanticCache::new(Arc::new(pool.clone()), embedder));

    let connector_manager = Arc::new(ConnectorManager::new(Arc::clone(&circuit_breaker)));

    // JSONL telemetry side-channel — see src/telemetry.rs. Separate from
    // the Postgres request_logs audit trail; useful for local ROI reports
    // without a DB round trip, and survives even if Postgres is down.
    let telemetry_dir = std::env::var("TELEMETRY_OUTPUT_DIR").unwrap_or_else(|_| "./telemetry".to_string());
    let telemetry_buffer_size: usize = std::env::var("TELEMETRY_BUFFER_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let telemetry = Arc::new(
        TelemetryRecorder::new(&telemetry_dir, telemetry_buffer_size)
            .expect("failed to initialize telemetry recorder — check TELEMETRY_OUTPUT_DIR is writable"),
    );

    info!("Components initialized");

    let max_concurrent_provider_calls: usize = std::env::var("MAX_CONCURRENT_PROVIDER_CALLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let concurrency_limiter = Arc::new(ConcurrencyLimiter::new(max_concurrent_provider_calls));
    info!(max_concurrent_provider_calls, "Concurrency limiter configured");

    let state = AppState {
        route_engine,
        connector_manager,
        cost_tracker,
        circuit_breaker,
        semantic_cache,
        rate_limiter: Arc::clone(&rate_limiter),
        loop_guard,
        spend_guard,
        telemetry,
        concurrency_limiter,
    };

    let protected_routes = Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(
            api_key_store,
            api_key_middleware,
        ));

    let public_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/audit/daily", get(audit_daily_handler))
        .with_state(state);

    // Admin dashboard — see src/admin.rs. Guarded by ROUTERFUEL_ADMIN_KEY
    // (X-Admin-Key header), a separate secret from per-client BYOK/auth
    // keys, since a client key must never be able to see other clients'
    // spend. If ROUTERFUEL_ADMIN_KEY isn't set, admin_key_middleware refuses
    // every request with 503 rather than silently leaving the dashboard open.
    let admin_key = Arc::new(std::env::var("ROUTERFUEL_ADMIN_KEY").unwrap_or_default());
    if admin_key.is_empty() {
        warn!("ROUTERFUEL_ADMIN_KEY is not set — the admin dashboard API is disabled until it is");
    } else {
        info!("Admin dashboard API enabled at /admin/* (X-Admin-Key required)");
    }

    let admin_state = AdminState {
        pool: Arc::new(pool.clone()),
        rate_limiter: Arc::clone(&rate_limiter),
    };

    let admin_routes = Router::new()
        .route("/admin/overview", get(admin::overview_handler))
        .route("/admin/cache", get(admin::cache_stats_handler))
        .route("/admin/models/expensive", get(admin::top_expensive_models_handler))
        .route("/admin/models/usage", get(admin::model_usage_handler))
        .route("/admin/clients", get(admin::client_spend_handler))
        .route("/admin/timeline", get(admin::timeline_handler))
        .route("/admin/rate-limits", get(admin::rate_limits_handler))
        .route("/admin/shadow", get(admin::shadow_stats_handler))
        .with_state(admin_state)
        .layer(middleware::from_fn_with_state(admin_key, admin::admin_key_middleware));

    // Every sub-router above already resolved its own state via
    // `.with_state()`, so they're all `Router<()>` here and can be merged
    // freely even though they started out with three different state types
    // (AppState for two of them, AdminState for the third).
    let app = Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .layer(DefaultBodyLimit::max(1024 * 1024 * 10))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(60)));

    let addr = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let bind_addr = format!("{}:{}", addr, port);

    let listener = TcpListener::bind(&bind_addr).await?;

    info!("Server listening on http://{}", bind_addr);
    info!("OpenAI-compatible endpoint: POST /v1/chat/completions (BYOK header required per provider, or X-Openrouter-Api-Key)");

    axum::serve(listener, app).await?;

    Ok(())
}
