# RouterFuel

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A BYOK (Bring Your Own Key) AI gateway written in Rust. RouterFuel sits between your app and the LLM providers you already have keys for — Anthropic, OpenAI, Gemini, DeepSeek, xAI, Mistral, Qwen, Moonshot, Zhipu, Meta, and OpenRouter as a universal fallback — and adds the routing, cost tracking, caching, and safety nets you'd otherwise have to build yourself.

RouterFuel never holds a billable key of its own. Every request is billed to *your* provider account, using *your* key. RouterFuel's job is just to route it well, cache it when it can, and tell you what it cost.

## Features

- **Smart routing** — pick a model by name, let RouterFuel auto-select on cost/latency/quality, or route by task type (`task:summarize`, `task:code`, etc.)
- **BYOK across 11 providers** — supply your own key per provider via request headers; OpenRouter acts as a universal fallback if that's the only key you have
- **Vision support** — send images (URL or base64) to any vision-capable model in the registry
- **Semantic caching** — a local ONNX embedding model (no external API cost) matches semantically similar prompts and serves cached responses instead of re-calling a provider
- **Cost tracking & audit trail** — every request is logged with token counts, cost, latency, and savings vs. a GPT-4o baseline
- **Circuit breaker** — automatically stops sending traffic to a provider that's returning errors, and probes it back into rotation once it recovers
- **Rate limiting & tiers** — per-client rate limits (free / pro / enterprise), configurable via env var or a Postgres table; tier changes take effect on the **next server restart** (tiers are loaded once at startup, not watched live)
- **Concurrency limiting** — bounds in-flight provider calls so a traffic spike doesn't get you rate-limited or IP-blocked upstream
- **Guardrails** — LoopGuard flags a client stuck retrying the same prompt; SpendGuard hard-caps per-client spend in a rolling window
- **Shadow-mode A/B testing** — fire a second, comparison-only call at a different model alongside the real one, without affecting what the client receives. **Enabled by default** — any client can trigger it by sending `shadow_model` on a request, and it bills a second real call to their BYOK key
- **Streaming** — full SSE streaming support for Anthropic, Gemini, and every OpenAI-compatible provider
- **Admin dashboard** — a self-hosted, no-build-step web UI at `/admin/dashboard` visualizing spend, cache performance, per-model and per-client cost, the request timeline, rate-limit tiers, and shadow-mode comparisons — reads the `/admin/*` endpoints below in real time. The dashboard *page* itself is public; the data endpoints it calls each require `X-Admin-Key`
- **Cursor integration** — point Cursor's custom OpenAI-compatible model settings straight at RouterFuel and route your editor's requests through your own provider keys

## Requirements

- Rust (2021 edition)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- A local ONNX sentence-embedding model + tokenizer (e.g. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) for semantic caching — download the `.onnx` and `tokenizer.json` files and point RouterFuel at them (see below)

## Setup

**1. Clone and build**

```
git clone https://github.com/uaz5/Routerfuel.git
cd Routerfuel
cargo build --release
```

**2. Set up the database**

Create a Postgres database with the `vector` extension available, then run the migrations in `migrations/` in order (001 through 006). If you're using `sqlx-cli`:

```
sqlx migrate run
```

Migrations run automatically on startup too, via `sqlx::migrate!` in `main.rs`.

**3. Set environment variables**

| Variable                        | Required | Default       | Purpose                                                             |
| -------------------------------- | -------- | ------------- | -------------------------------------------------------------------- |
| `DATABASE_URL`                  | yes      | —             | Postgres connection string                                          |
| `ROUTERFUEL_API_KEYS`           | no       | empty         | Client API keys, format `sha256hex:ClientName,sha256hex:ClientName` |
| `ROUTERFUEL_CLIENT_TIERS`       | no       | empty         | Per-client rate tiers, format `raw_key:pro,raw_key:enterprise`      |
| `ROUTERFUEL_ADMIN_KEY`          | no       | empty         | Key required to access `/admin/*` endpoints (`X-Admin-Key` header)  |
| `EMBEDDING_MODEL_PATH`          | no       | `./models/embedding.onnx` | Path to your local ONNX embedding model (enables semantic cache) |
| `EMBEDDING_TOKENIZER_PATH`      | no       | `./models/tokenizer.json` | Path to the matching tokenizer.json                       |
| `LOOP_GUARD_REPEAT_THRESHOLD`   | no       | 4             | Repeats of an identical prompt before it's flagged as a loop        |
| `LOOP_GUARD_WINDOW_SECS`        | no       | 60            | Window LoopGuard checks over                                        |
| `MAX_SPEND_CENTS_PER_CLIENT`    | no       | 5000          | Per-client spend cap (cents) per window                             |
| `SPEND_GUARD_WINDOW_SECS`       | no       | 3600          | SpendGuard rolling window, in seconds                               |
| `MAX_CONCURRENT_PROVIDER_CALLS` | no       | 200           | Caps simultaneous in-flight provider calls                          |
| `ENABLE_SHADOW_MODE`            | no       | **true**      | Enables shadow-mode A/B comparison calls — on by default; set to `false` to disable |
| `TELEMETRY_OUTPUT_DIR`          | no       | `./telemetry` | Where telemetry JSONL files are written                             |
| `TELEMETRY_BUFFER_SIZE`         | no       | 500           | Records buffered before a telemetry flush                           |
| `HOST`                          | no       | `0.0.0.0`     | Bind address                                                        |
| `PORT`                          | no       | `3000`        | Bind port                                                           |

To generate an API key hash for `ROUTERFUEL_API_KEYS`:

```
echo -n "rf_live_yoursecretkey" | sha256sum | awk '{print $1}'
```

**4. Run it**

```
cargo run --release
```

RouterFuel is now listening on `http://localhost:3000` (or whatever `HOST`/`PORT` you set).

See [USAGE.md](https://github.com/uaz5/Routerfuel/blob/main/USAGE.md) for how to actually call it, including the admin dashboard UI and Cursor setup.

## Project structure

```
src/
  main.rs                 — HTTP server, routing glue, request handlers
  connectors.rs            — per-provider HTTP clients (Anthropic, Gemini, OpenAI-compatible)
  route_engine.rs           — model registry + routing decisions
  auth.rs                   — API key validation, BYOK header extraction, Cursor composite-key bridge
  rate_limiter.rs           — per-client tiered rate limiting
  client_registry.rs        — loads client tiers from env/Postgres
  circuit_breaker.rs        — per-provider health tracking
  concurrency.rs            — bounds in-flight provider calls
  guardrails.rs             — LoopGuard + SpendGuard
  semantic_cache.rs         — pgvector-backed semantic cache
  embedder.rs               — local ONNX embedding model
  vision.rs                 — multimodal message types + per-provider image formatting
  tokens.rs                 — tiktoken-based token counting
  cost_tracker.rs           — request logging + cost/savings reports
  telemetry.rs              — JSONL telemetry + ROI reports
  streaming.rs              — SSE streaming handler
  admin.rs                  — /admin/* dashboard data endpoints, incl. /audit/daily
  openrouter_catalog.rs     — pulls OpenRouter's public model list into the registry
static/
  dashboard.html            — self-contained admin dashboard UI, served at /admin/dashboard
migrations/                — Postgres schema, run in numeric order
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](https://github.com/uaz5/Routerfuel/blob/main/LICENSE) file for details.
