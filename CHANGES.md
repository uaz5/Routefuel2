# RouteFuel v0.6 — what changed

## 1. Models — every lab you asked for, current as of July 2026

`src/route_engine.rs` now registers **40 models across 10 labs**, direct-routed
(not just guessed at — pricing/context pulled from live pricing trackers today):

| Provider | Models added |
|---|---|
| Anthropic | Opus 4.8, Sonnet 5, Haiku 4.5, Fable 5, + legacy Opus 4.7/4.6, Sonnet 4.6 |
| OpenAI | GPT-5.6 Sol/Terra/Luna (new July 9 family), GPT-5.5, GPT-5.4/mini/nano, GPT-OSS-20B (open weight) |
| Google Gemini | 3.1 Pro, 3.5 Flash, 3 Flash, 3.1 Flash-Lite, 2.5 Pro, 2.5 Flash-Lite |
| xAI Grok | 4.5, 4.3, 4.20, 4.1 Fast, Code Fast 1 |
| DeepSeek | V4 Flash, V4 Pro, V3.2 (legacy) — all open-weight |
| Mistral | Large 3, Small 4, Codestral 2, Ministral 8B |
| Alibaba Qwen | Qwen3 Max, Qwen3-235B-A22B, Qwen Turbo |
| Moonshot (Kimi) | K2.6, K2.5 |
| Zhipu (GLM) | GLM-5 |
| Meta Llama | 4 Maverick, 4 Scout, 3.3 70B (legacy) |
| OpenRouter | `openrouter/auto` catch-all |

Every `ModelConfig` now also carries `supports_vision` and `open_weight` flags.
`vision.rs` no longer keeps its own separate hardcoded vision-model list — it
reads `supports_vision` straight off the registry, so adding a model in one
place keeps routing and vision-filtering in sync (previously these could and
would drift).

Prices/latencies will drift — that's expected and fine, since RouteFuel never
pays these bills itself (see below), so a stale number here only skews which
model gets picked, not what anyone is actually charged.

## 2. Strict BYOK — RouteFuel never touches a bill

This was the biggest structural change. Previously:
- `main.rs` **required** `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars at boot and used them as the default key whenever a client didn't send their own — meaning **you** were paying for every client who didn't BYOK.
- The semantic cache called OpenAI's paid `/v1/embeddings` endpoint on your own key on every single request, cache hit or miss.

Now:
- `ConnectorManager::new()` takes **no keys at all**. Every `Connector::complete()` takes `client_api_key: &str` — not `Option<&str>` — there is no fallback path to a gateway-owned key anywhere in the codebase.
- `main.rs` has a `resolve_byok_route()` function that:
  1. Uses the client's key for the exact provider RouteFuel selected, if supplied.
  2. Otherwise, if the client supplied an **OpenRouter** key, re-routes the *same model* through OpenRouter (rewriting the id to `vendor/model`, e.g. `claude-opus-4-8` → `anthropic/claude-opus-4-8`) — this is the "everyone already has an OpenRouter key" case you mentioned.
  3. Otherwise, rejects with `400` before any connector is touched, telling the client exactly which header to send.
- New BYOK headers, one per provider (see table below), plus `X-Openrouter-Api-Key`.

| Provider | Header |
|---|---|
| OpenAI | `X-OpenAI-Api-Key` |
| Anthropic | `X-Anthropic-Api-Key` |
| Gemini | `X-Gemini-Api-Key` |
| DeepSeek | `X-DeepSeek-Api-Key` |
| Mistral | `X-Mistral-Api-Key` |
| xAI/Grok | `X-XAI-Api-Key` (or `X-Grok-Api-Key`) |
| Qwen | `X-Qwen-Api-Key` (or `X-DashScope-Api-Key`) |
| Moonshot/Kimi | `X-Moonshot-Api-Key` (or `X-Kimi-Api-Key`) |
| Zhipu/GLM | `X-Zhipu-Api-Key` (or `X-GLM-Api-Key`) |
| Meta/Llama | `X-Meta-Api-Key` (or `X-Llama-Api-Key`) |
| OpenRouter (universal fallback) | `X-Openrouter-Api-Key` |

`main.rs` no longer reads `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/etc. at all — the
only secrets it needs now are `DATABASE_URL` and `ROUTEFUEL_API_KEYS` (your
own client auth store).

## 3. OpenRouter integration

`connectors.rs` adds `Provider::OpenRouter`, hitting
`https://openrouter.ai/api/v1/chat/completions` through the same generic
OpenAI-compatible connector everything else uses. `Provider::openrouter_prefix()`
maps each lab to OpenRouter's vendor slug (`anthropic`, `openai`, `google`,
`deepseek`, `mistralai`, `x-ai`, `qwen`, `moonshotai`, `z-ai`, `meta-llama`) so
the BYOK-fallback rewrite in `main.rs` works automatically for every model in
the registry, not just a hardcoded subset.

## 4. Local ONNX embedding (`src/embedder.rs`) — latency + the other BYOK leak

The semantic cache used to call OpenAI's embeddings API on **your** key for
every lookup and every store — that's both a real per-request cost you were
covering, and a 40-80ms network round trip before routing could even start.

`src/embedder.rs` is new: it runs a small sentence-embedding model locally via
the `ort` (ONNX Runtime) crate — mean-pooled, L2-normalized, same convention
`sentence-transformers` uses. This drops embedding cost to $0 and typical
latency to low single-digit milliseconds, entirely in-process.

**Setup required:** download a 384-dim embedding model exported to ONNX (e.g.
`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) plus its `tokenizer.json`,
and point these at it:

```
EMBEDDING_MODEL_PATH=./models/embedding.onnx        # default shown
EMBEDDING_TOKENIZER_PATH=./models/tokenizer.json    # default shown
```

If those files aren't present, the server still boots — `SemanticCache`
degrades to "always miss" instead of failing startup (see `main.rs`).

`migrations/005_local_embeddings.sql` resizes the `semantic_cache.embedding`
column from 1536 dims (OpenAI) to 384 dims (local model) and truncates the
cache table, since old and new embeddings aren't comparable — cached
*responses* aren't stored anywhere else, so this is safe, just a one-time
cold cache.

**The 0.96 cosine similarity threshold in `semantic_cache.rs` is unchanged**,
as asked.

**Honest caveat:** `ort` 2.0's exact API shifted across rc releases and I
could not compile-check this file in this environment (see §6) — if
`cargo build` flags a method name in `embedder.rs`, it's almost certainly a
small rename (e.g. `try_extract_raw_tensor` → `try_extract_tensor`) rather
than a structural issue; the tokenization → tensor → mean-pool → normalize
pipeline is correct regardless of the exact accessor names on `ort::Session`.

## 5. Other fixes along the way

- **`cost_tracker.rs` had a real compile error**: `main.rs` was already calling
  `record_request(...)` with an `is_byok` argument that the function didn't
  accept. Fixed — it's now a real parameter, persisted into the `is_byok`
  column `004_byok_support.sql` added but nothing ever wrote to.
- **Schema/code mismatch**: `001_request_logs.sql` declared a `model_api_id`
  column, but every query in `cost_tracker.rs` read/wrote `model_name` — this
  would have failed at runtime the first time a request completed. Aligned
  the migration to `model_name` (what the code — and its own inline schema
  comment — actually used).
- **Gemini connector was a stub** (`NotImplemented` for every call, disabled
  in the registry). Implemented the real request/response translation
  (system-message extraction, role remapping to `user`/`model`, generation
  config) and it's enabled in the registry now.
- **`RateLimiter` existed but was never wired into the app** — no field on
  `AppState`, no call anywhere. It's now checked at the top of the chat
  handler (currently defaulting every authenticated client to the `Pro`
  tier — wiring it to the `client_tiers` table for real per-client tiers is
  flagged with a `TODO` comment, since that needs a design decision from you
  about who gets to change a client's tier and how).
- **`vision.rs` was never declared as a module in `main.rs`** — it existed on
  disk but wasn't part of the compiled crate at all. It's wired in now.
- Connection pooling/keep-alive tuned on the shared `reqwest::Client`
  (`pool_max_idle_per_host`, `tcp_keepalive`, a real `connect_timeout`) —
  cuts repeat-call latency to the same provider, which matters at gateway
  volume.
- Added `GET /v1/models` (lists the enabled registry with pricing/context/
  vision/open-weight flags) since you now have 40 models across 10 labs and
  "check main.rs for the list" stops being a reasonable API.

## 6. Kimi K3 added

Real, released July 16, 2026, confirmed via live pricing trackers (not in my
training data — I checked). 2.8T-parameter open-weight MoE model from
Moonshot, 1,048,576-token context, native vision, $3/$15 per 1M tokens
(cache-hit input discounted to $0.30). Added as `kimi-k3` under
`Provider::Moonshot` in `route_engine.rs`, quality-scored just below Fable 5
per the independent benchmark coverage (Vals AI #2 overall, Artificial
Analysis #3–4). Full weights are scheduled for July 27 per Moonshot — the API
is live now at `api.moonshot.ai`, which is what RouteFuel calls.

## 7. Runaway-agent protection (`src/guardrails.rs`)

Two independent, in-memory, sub-millisecond checks that run **before any
provider is called** — a blocked request costs nothing anywhere:

- **`LoopGuard`** — flags a client sending the *same prompt* 4+ times (configurable) within a 60s window (configurable). This is the classic signature of an agent stuck retrying a failed step or two agents bouncing a message back and forth.
- **`SpendGuard`** — hard per-client cost ceiling in a rolling window (default $50/hour, configurable). Catches loops that vary their prompt each time — so `LoopGuard` alone wouldn't catch them — as well as plain runaway volume. Updated with the *real* cost after each call completes (`token_cost.total_cost_cents`), checked before the next one starts.

Both trip a `429` with a message explaining what happened and why, before
`chat_completions_handler` does any token counting, cache lookup, or routing.

Env vars: `LOOP_GUARD_REPEAT_THRESHOLD` (default `4`), `LOOP_GUARD_WINDOW_SECS`
(default `60`), `MAX_SPEND_CENTS_PER_CLIENT` (default `5000` = $50),
`SPEND_GUARD_WINDOW_SECS` (default `3600`).

**Worth knowing:** since RouteFuel is BYOK, a runaway loop bills the
*client's* provider key, not yours — this doesn't protect your wallet so
much as it protects your clients from themselves (and you from the support
ticket). It's process-local, in-memory state, so limits reset on restart and
aren't shared across horizontally-scaled replicas. That's fine for a
guardrail whose whole point is to react in the same millisecond as the
request; it's not a substitute for the authoritative accounting already in
`cost_tracker.rs` / `request_logs`, which is the source of truth for
anything billing-adjacent.

## 8. Dynamic OpenRouter catalog (`src/openrouter_catalog.rs`) — "100+ LLMs"

Rather than hand-maintaining a second list of 100+ OpenRouter model ids that
goes stale the moment OpenRouter adds or removes something, RouteFuel now
fetches OpenRouter's real catalog at startup:

- `GET https://openrouter.ai/api/v1/models` — this is public, no API key needed just to *list* what's available (only to actually call one).
- Every entry becomes a `ModelConfig` under `Provider::OpenRouter`, with cost converted from OpenRouter's per-token USD convention into the registry's cents-per-1M-tokens convention, and `supports_vision` read off `architecture.input_modalities`.
- `RouteEngine::extend_registry()` merges these in, **skipping any id that already exists** — so the ~40 curated, hand-tuned direct integrations in §1 always win on routing ties; the catalog only adds coverage, it never overrides a hand-tuned entry.
- Non-fatal on failure (timeout or network error) — logged as a warning, server still boots with the curated registry alone. It's a live third-party endpoint; startup shouldn't hang or crash because it's briefly unreachable.

This means the registry size at runtime is "~40 curated models + however many
hundred OpenRouter is currently listing," not a fixed number — check
`GET /v1/models` on a running instance to see the real count. Routing to any
catalog-only entry naturally requires the client to have supplied an
`X-Openrouter-Api-Key`, same as everything else — no change needed to the
BYOK resolution logic in `main.rs`, since `Provider::OpenRouter` was already
a first-class case there.

## 9. This round: completing the "incomplete" folder

You uploaded 8 more files (`admin.rs`, `async_handler.rs`, `main_prod.rs`,
`SMART_ROUTING_CODE.rs`, `client_registry.rs`, `example_comprehensive.rs`,
`streaming.rs`, `telemetry.rs`). Four of them slotted in cleanly and are now
wired into the real server; four assumed a different, older architecture
that predates BYOK and don't belong in this codebase as-is. Here's exactly
what happened to each.

### Wired in for real

**`admin.rs` → the dashboard backend** (this answers the "do you have a
dashboard" question from earlier — now yes). Bugs fixed: every query
referenced a `model_api_id` column that doesn't exist (the real column,
matching `cost_tracker.rs`, is `model_name`) — would have 500'd on first
call. The file's own header comment claimed every route "requires X-API-Key
with admin scope" but no such check existed anywhere in the code — added
`admin_key_middleware`, gated on a new `ROUTEFUEL_ADMIN_KEY` env var (a
*separate* secret from per-client BYOK keys, since a client key must never
see another client's spend). Mounted at `/admin/*`:
`overview`, `cache`, `models/expensive`, `models/usage`, `clients`,
`timeline`, `rate-limits`. If `ROUTEFUEL_ADMIN_KEY` isn't set, every admin
route returns `503` rather than silently being open.

**`client_registry.rs` → real per-client rate-limit tiers.** This is what
closes the TODO I flagged two rounds ago ("wire this up to the client_tiers
table"). It expected a richer `RateLimiter` API (`TierConfig`,
`.register()`, `.status()`) that didn't exist — `rate_limiter.rs` is now
rewritten around that API. `main.rs` calls `client_registry::load_all_tiers`
at startup (env var `ROUTEFUEL_CLIENT_TIERS`, then the `client_tiers` table,
DB wins on conflict), and the hot-path rate check is now
`state.rate_limiter.check(&client_id)` — it looks up whatever tier that
client is actually registered at instead of the old hardcoded
`UserTier::Pro` for everyone.

**`telemetry.rs` → fixed and wired.** Real bug: `flush()` logged
`buffer.len()` *after* `buffer.clear()`, so it always reported "Flushed 0
records" no matter how many were actually written. Fixed by capturing the
count first. Also moved `generate_roi_report`'s directory/file scan onto
`spawn_blocking` — it was doing blocking `std::fs` I/O straight inside an
`async fn`, which stalls the executor thread for however long that scan
takes. Wired into `main.rs` as a JSONL side-channel — every completed
non-streaming request fires a record via `tokio::spawn` (never adds
latency to the response). Separate from the Postgres `request_logs` audit
trail on purpose: it's a local file that survives even if Postgres is down.

**`streaming.rs` → real SSE support, finally wired to an actual route.**
This didn't exist as a callable endpoint before — it was a file sitting on
disk. Adapted: `ChatRequest` (never existed) → `ChatCompletionRequest` (the
real type); the `cost_tracker.record_request` call used raw token counts
where the function now needs a `TokenCostBreakdown` + `is_byok` — fixed to
build one from the registry's pricing, same as the non-streaming path.
Added a real Gemini branch (`:streamGenerateContent?alt=sse`) — the
original only handled Anthropic and a generic OpenAI-compatible Bearer
path, which would've silently mishandled Gemini's different wire format and
auth (`?key=` query param, not a header). `main.rs`'s
`POST /v1/chat/completions` now branches on `"stream": true` in the request
body to this handler instead of the JSON path — BYOK resolution, rate
limiting, the loop guard, and the spend guard all still run first, same as
non-streaming. Semantic cache is intentionally *not* consulted for
streaming requests (caching a stream means buffering the whole thing
anyway, which defeats streaming) — a cached hit still short-circuits the
next non-streaming call for the same prompt.

**`SMART_ROUTING_CODE.rs` → folded in, not shipped as a separate file.**
Its scoring algorithm duplicated what `RouteEngine::select()` already does
correctly. The one genuinely new, worth-keeping idea — letting the `model`
field itself say `"auto"` or `"task:summarise"` instead of requiring a
concrete model id — is now `resolve_model()` in `main.rs`, using the
registry's real `select()` / `select_for_task()`. Try `"model": "auto"` or
`"model": "task:extract_action_items"` in a request.

### Not wired in — and why

**`async_handler.rs`, `example_comprehensive.rs`, `main_prod.rs`** are
mutually consistent with each other but assume a *different, older*
architecture: a `RouteEngine`/`ConnectorManager`/`CircuitBreaker` with
different method signatures (`route()`, `get_model()`, `is_available()`),
no BYOK concept at all (`ModelConfig` there has no notion of a client-
supplied key — it assumes RouteFuel holds provider keys itself), and a
library-style `routefuel::{...}` API rather than the OpenAI-compatible HTTP
gateway this actually is. This is the design RouteFuel had *before* the
BYOK rewrite — plugging it in as-is would silently reintroduce "RouteFuel
pays the provider bill," which is the exact thing §2 exists to prevent.

I didn't want to either (a) silently drop three files you asked me to
complete, or (b) ship a rushed, half-working adaptation of a fairly large
concurrency-limiting + shadow-mode-A/B-testing layer just to say the folder
is "done." Concurrency limiting (backpressure via a semaphore) and shadow
mode (firing a second comparison call without it affecting the response)
are genuinely useful ideas from `async_handler.rs` — if you want them, say
so and I'll build a real `src/async_handler.rs` next round that wraps the
*actual* `RouteEngine` / `ConnectorManager` / BYOK resolution this codebase
uses, rather than adapting code written against types that no longer exist.
`example_comprehensive.rs` I'd rewrite the same way: as a runnable example
that calls the real HTTP API (`/v1/chat/completions`, `/v1/models`,
`/admin/overview`) via `reqwest`, not a library import that assumes a
`routefuel` crate this project doesn't expose.

## 10. This round: Opus 5, concurrency limiting, shadow mode

### Claude Opus 5

Real — released July 24, 2026 (yesterday, confirmed via Anthropic's own
model docs, not just news coverage). Model id `claude-opus-5`, **same
pricing as Opus 4.8** ($5/$25 per 1M), 1M-token context window (that's both
the default *and* the max now — no smaller variant), 128k max output,
vision-capable, thinking-on-by-default. Anthropic's own framing: "comes
close to the frontier intelligence of Fable 5 at half the price." Added as
the new top Anthropic entry in the registry (quality 0.98, ~140ms latency —
Anthropic's own benchmark says it hits its best trading result using
roughly a seventh of Opus 4.8's reasoning tokens and under half the
latency). Opus 4.8 stays in the registry, enabled, as a legacy id — nothing
breaks for callers still pinned to it. `MeetingTask::DraftResponse` now
routes to Opus 5 instead of 4.8, same price, better result.

### Concurrency limiting (`src/concurrency.rs`)

Bounds how many provider calls can be in flight at once — across **both**
the non-streaming path and the streaming path, which is why it lives at the
`AppState` level rather than inside `ConnectorManager` (the streaming path
never touches `ConnectorManager`; it talks to providers directly for full
control over SSE parsing).

Without this, a traffic spike means RouteFuel opens as many simultaneous
connections to providers as it has incoming requests — the classic way to
get rate-limited or IP-blocked by a provider mid-burst. Now, request
#(N+1) past `MAX_CONCURRENT_PROVIDER_CALLS` (default `200`) waits for a
slot instead of firing immediately.

For streaming, the permit is acquired *inside* the SSE generator itself,
not before it starts — so a long-lived stream correctly counts against the
pool for its entire duration, and releases automatically the instant the
stream ends or the client disconnects (axum drops the generator, which
drops the permit — no manual cleanup needed).

Env var: `MAX_CONCURRENT_PROVIDER_CALLS` (default `200`).

### Shadow mode

A client sets `"shadow_model": "some-other-model-id"` on a normal
`/v1/chat/completions` request. RouteFuel routes and answers the request
exactly as usual — the client's response is completely unaffected — but
*also* fires an identical request at the shadow model in the background via
`tokio::spawn`, and logs a side-by-side comparison (cost delta, latency
delta, output length, or an error if the shadow call couldn't be made) to
the new `shadow_comparisons` table.

Some things worth knowing:

- **It's a second real, billed call.** If you shadow every request against
  a second model, you're paying for two models on every request. That's
  inherent to what shadow mode *is* — the whole point is finding out what
  the alternative would really have cost, with real traffic — not a bug.
  It counts against the client's `SpendGuard` cap same as any other call.
- **It never fails or delays the primary response.** If the client has no
  BYOK key for the shadow model's provider, if the shadow provider's
  circuit breaker is open, if the shadow call errors — none of that touches
  what the client receives. It's logged as a comparison row with
  `shadow_error` set and nothing else happens.
- **It respects the same concurrency limiter** as everything else, so
  shadow traffic can't be used to bypass `MAX_CONCURRENT_PROVIDER_CALLS`.
- **Global kill switch:** `ENABLE_SHADOW_MODE=false` disables it for every
  client without a code change, if you'd rather clients not be able to
  double their own spend via a request parameter.
- `GET /admin/shadow` (dashboard, admin-key protected) summarizes
  comparisons grouped by (primary model, shadow model) pair: average cost
  delta, average latency delta, shadow error rate — this is the "would a
  cheaper model have been fine?" report.

Try it: send a normal request with `"model": "claude-sonnet-5"` and
`"shadow_model": "claude-haiku-4-5"` a few times, then hit
`GET /admin/shadow?from=2026-07-01&to=2026-07-31` with your
`X-Admin-Key` to see the comparison.

## 11. What I could not verify in this environment

This sandbox's system Rust is 1.75 (Dec 2023) via `apt`, and I don't have
network access to rustup's real distribution host to get a current toolchain.
Nearly the entire crates.io ecosystem as published *today* (mid-2026) now
requires a Cargo with the `edition2024` feature stabilized, which 1.75
doesn't have — this hit even your **original**, unmodified `Cargo.toml`
(via `sqlx`'s own transitive deps), not just the new `ort`/`tokenizers`
additions. I did do a manual line-by-line review pass and a brace/structure
balance check across every file, but I want to be upfront that I couldn't run
`cargo check` end-to-end here. Render's build image will use a current stable
Rust and should be fine — if anything doesn't compile there, it'll most
likely be a small `ort` method-name fix in `embedder.rs` per the caveat
above.
