-- ============================================================================
-- 006_shadow_comparisons.sql — RouterFuel v0.8
--
-- Shadow mode: when a client sets `shadow_model` on a request, RouterFuel
-- fires an identical call at that model *in addition to* the normally-
-- routed one, purely for comparison — the client only ever sees the primary
-- response. This table is where those comparisons land. See
-- main.rs::maybe_fire_shadow_request and CHANGES.md.
-- ============================================================================

CREATE TABLE IF NOT EXISTS shadow_comparisons (
    id                      BIGSERIAL PRIMARY KEY,
    request_id              VARCHAR(36)     NOT NULL,
    client_id               VARCHAR(64),

    primary_model           VARCHAR(100)    NOT NULL,
    primary_provider        VARCHAR(50)     NOT NULL,
    primary_cost_cents      DOUBLE PRECISION NOT NULL,
    primary_latency_ms      INT             NOT NULL,
    primary_output_chars    INT             NOT NULL,

    shadow_model            VARCHAR(100)    NOT NULL,
    shadow_provider         VARCHAR(50)     NOT NULL,
    -- Nullable: the shadow call can fail (provider error, no BYOK key for
    -- that provider, circuit open) without ever touching the primary
    -- response the client received — a NULL here just means "shadow call
    -- didn't complete," logged via shadow_error instead.
    shadow_cost_cents       DOUBLE PRECISION,
    shadow_latency_ms       INT,
    shadow_output_chars     INT,
    shadow_error            TEXT,

    cost_delta_cents        DOUBLE PRECISION, -- shadow_cost - primary_cost (negative = shadow cheaper)
    latency_delta_ms        INT,              -- shadow_latency - primary_latency

    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shadow_request       ON shadow_comparisons (request_id);
CREATE INDEX IF NOT EXISTS idx_shadow_client         ON shadow_comparisons (client_id);
CREATE INDEX IF NOT EXISTS idx_shadow_models         ON shadow_comparisons (primary_model, shadow_model);
CREATE INDEX IF NOT EXISTS idx_shadow_created        ON shadow_comparisons (created_at);

COMMENT ON TABLE shadow_comparisons IS
    'A/B comparisons from shadow-mode requests: primary model actually served to the client vs. a shadow model called in parallel purely for comparison.';
