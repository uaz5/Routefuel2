-- =============================================================================
-- RouterFuel v0.4  —  Request logs audit table
-- Run automatically by sqlx::migrate! on startup
-- =============================================================================

CREATE TABLE IF NOT EXISTS request_logs (
    id                  BIGSERIAL PRIMARY KEY,

    -- Identification
    request_id          VARCHAR(36)     NOT NULL UNIQUE,
    provider            VARCHAR(20)     NOT NULL,
    model_name        VARCHAR(100)    NOT NULL,

    -- Token counts (from tiktoken-rs precision counting)
    tokens_in           INTEGER         NOT NULL DEFAULT 0,
    tokens_out          INTEGER         NOT NULL DEFAULT 0,

    -- Costs in USD cents (6 decimal places for sub-cent accuracy)
    cost_cents          DECIMAL(14,6)   NOT NULL DEFAULT 0,
    cost_saved_cents    DECIMAL(14,6)   NOT NULL DEFAULT 0,

    -- Performance
    latency_ms          INTEGER         NOT NULL DEFAULT 0,
    routing_decision_ms INTEGER         NOT NULL DEFAULT 0,

    -- Client metadata
    client_id           VARCHAR(100),
    user_tier           VARCHAR(20),
    priority            VARCHAR(20),

    -- Outcome
    status              VARCHAR(20)     NOT NULL DEFAULT 'success',
    error_message       TEXT,

    -- Timestamp (immutable)
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Indexes for the queries used in audit reports
CREATE INDEX IF NOT EXISTS idx_rl_request_id   ON request_logs (request_id);
CREATE INDEX IF NOT EXISTS idx_rl_provider      ON request_logs (provider);
CREATE INDEX IF NOT EXISTS idx_rl_model         ON request_logs (model_name);
CREATE INDEX IF NOT EXISTS idx_rl_client        ON request_logs (client_id);
CREATE INDEX IF NOT EXISTS idx_rl_created_at    ON request_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_rl_status        ON request_logs (status);

--- Composite index for daily report queries
CREATE INDEX IF NOT EXISTS idx_rl_date_provider
    ON request_logs (created_at, provider)
    WHERE status = 'success';

COMMENT ON TABLE  request_logs                IS 'Immutable audit trail of every request routed through RouterFuel';
COMMENT ON COLUMN request_logs.cost_cents     IS 'Actual cost in USD cents for this request';
COMMENT ON COLUMN request_logs.cost_saved_cents IS 'Savings vs GPT-4o baseline in USD cents';
COMMENT ON COLUMN request_logs.routing_decision_ms IS 'Time to select a model — should be <10ms';
