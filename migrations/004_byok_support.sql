-- =============================================================================
-- migrations/004_byok_support.sql — RouteFuel v0.5
-- Adds is_byok column to track custom key requests vs gateway-billed requests
-- =============================================================================

ALTER TABLE request_logs 
    ADD COLUMN IF NOT EXISTS is_byok BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_rl_is_byok 
    ON request_logs (is_byok) 
    WHERE is_byok = TRUE;

COMMENT ON COLUMN request_logs.is_byok IS 'TRUE if request used custom client provider API key (Bring Your Own Key)';