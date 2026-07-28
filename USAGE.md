# Using RouterFuel

This assumes RouterFuel is already running (see [README.md](README.md) for setup). Everything below assumes it's reachable at `http://localhost:3000` — swap in your real host.

## 1. Authenticate

Every request needs a RouterFuel API key in the `X-API-Key` header. This is *not* the same as your OpenAI/Anthropic/etc. key — it's a key you issue yourself via `ROUTERFUEL_API_KEYS`.

```bash
curl http://localhost:3000/health
# no auth needed for /health

curl http://localhost:3000/v1/chat/completions \
  -H "X-API-Key: rf_live_yoursecretkey" \
  ...
```

An invalid or missing key returns `401 Unauthorized`.

## 2. Bring your own provider key(s)

RouterFuel doesn't hold billable provider keys — you supply yours as headers on each request. Only supply the header(s) for the provider(s) you want to use:

| Provider | Header |
|---|---|
| OpenAI | `X-OpenAI-API-Key` |
| Anthropic | `X-Anthropic-API-Key` |
| Gemini | `X-Gemini-API-Key` |
| DeepSeek | `X-DeepSeek-API-Key` |
| Mistral | `X-Mistral-API-Key` |
| xAI / Grok | `X-XAI-API-Key` (or `X-Grok-API-Key`) |
| Qwen | `X-Qwen-API-Key` (or `X-DashScope-API-Key`) |
| Moonshot / Kimi | `X-Moonshot-API-Key` (or `X-Kimi-API-Key`) |
| Zhipu / GLM | `X-Zhipu-API-Key` (or `X-GLM-API-Key`) |
| Meta / Llama | `X-Meta-API-Key` (or `X-Llama-API-Key`) |
| OpenRouter | `X-OpenRouter-API-Key` |

If you only have an OpenRouter key, RouterFuel will route *any* model through OpenRouter automatically — you don't need a separate key per lab.

## 3. Send a chat completion

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "X-API-Key: rf_live_yoursecretkey" \
  -H "X-Anthropic-API-Key: sk-ant-yourkey" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-5",
    "messages": [
      { "role": "user", "content": "Explain what a circuit breaker does in one sentence." }
    ]
  }'
```

The response shape matches the OpenAI chat completions format:

```json
{
  "choices": [
    { "index": 0, "message": { "role": "assistant", "content": "..." }, "finish_reason": "stop" }
  ],
  "usage": { "prompt_tokens": 12, "completion_tokens": 20, "total_tokens": 32 }
}
```

### Picking a model

The `model` field accepts three forms:

- **A specific model ID** — e.g. `"claude-sonnet-5"`, `"gpt-5.6-sol"` — routes directly to that model/provider
- **`"auto"`** — RouterFuel picks the best model for you based on cost, latency, and quality, balanced by default
- **`"task:<name>"`** — routes based on task type, e.g. `"task:summarize"`, `"task:code"` — RouterFuel picks whichever registered model is best suited to that task

### Sending images (vision)

Use `content` as an array of parts instead of a plain string:

```json
{
  "model": "auto",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What's in this image?" },
        { "type": "image_url", "image_url": { "url": "https://example.com/photo.jpg" } }
      ]
    }
  ]
}
```

Or inline base64:

```json
{ "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,<...>" } }
```

If you use `"model": "auto"` or `"model": "task:..."`, RouterFuel automatically routes image-carrying requests to a vision-capable model. If you pin a specific model that doesn't support vision, you'll get a `400` telling you so — check `GET /v1/models` for which models support images.

### Streaming

Set `"stream": true` and read the response as Server-Sent Events:

```bash
curl -N http://localhost:3000/v1/chat/completions \
  -H "X-API-Key: rf_live_yoursecretkey" \
  -H "X-Anthropic-API-Key: sk-ant-yourkey" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-5",
    "stream": true,
    "messages": [{ "role": "user", "content": "Count to five." }]
  }'
```

### Shadow-mode A/B testing

Add a `shadow_model` field to compare a second model against the primary one, without the client ever seeing the shadow response:

```json
{
  "model": "claude-sonnet-5",
  "shadow_model": "gpt-5.6-sol",
  "messages": [{ "role": "user", "content": "Hello" }]
}
```

The comparison (cost delta, latency delta, output length) lands in the `shadow_comparisons` table and is queryable via `GET /admin/shadow`. Requires `ENABLE_SHADOW_MODE=true` on the server.

## 4. Other endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check, no auth |
| `/v1/models` | GET | List every model in the registry, with pricing/context window/vision support |
| `/v1/chat/completions` | POST | The main endpoint — see above |
| `/audit/daily` | GET | Daily cost/savings report |

### Admin dashboard

These require `X-API-Key` set to your `ROUTERFUEL_ADMIN_KEY`:

| Endpoint | Purpose |
|---|---|
| `/admin/overview` | Total spend, savings, latency for a date range |
| `/admin/cache` | Semantic cache hit rate and savings |
| `/admin/models/expensive` | Top 5 most expensive models by spend |
| `/admin/models/usage` | Usage breakdown by model |
| `/admin/clients` | Spend broken down per client |
| `/admin/timeline` | Day-by-day spend/request timeline |
| `/admin/rate-limits` | Current rate-limit tier per client |
| `/admin/shadow` | Shadow-mode A/B comparison stats |

Most take `start` and `end` query params (`YYYY-MM-DD`):

```bash
curl "http://localhost:3000/admin/overview?start=2026-07-01&end=2026-07-27" \
  -H "X-API-Key: your-admin-key"
```

## 5. Rate limits

Each client is assigned a tier — `free`, `pro`, or `enterprise` — via `ROUTERFUEL_CLIENT_TIERS` or the `client_tiers` Postgres table. A client with no explicit tier gets whatever the server's default is (falls back to `pro`). Exceeding your tier's requests-per-second returns `429`.

## 6. Troubleshooting

- **`401 Unauthorized`** — check your `X-API-Key` header is set and matches a hash in `ROUTERFUEL_API_KEYS`
- **`429 Too Many Requests`** — you've hit your tier's rate limit; wait or ask for a tier upgrade
- **`400` on an image request** — the model you pinned doesn't support vision; use `"model": "auto"` or check `/v1/models`
- **Slow first request** — the local embedding model and OpenRouter catalog fetch happen at startup, not per-request, so this shouldn't recur
