-- =============================================================================
-- migrations/003_client_tiers.sql  — RouteFuel v0.5
--
-- Stores per-client rate limit tiers so they can be changed without redeploy.
-- client_id = SHA-256 hex of the raw API key (same as auth store).
-- =============================================================================

CREATE TABLE IF NOT EXISTS client_tiers (
    client_id   VARCHAR(64)  PRIMARY KEY,
    tier        VARCHAR(20)  NOT NULL DEFAULT 'pro'
                             CHECK (tier IN ('free', 'pro', 'enterprise')),
    notes       TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Auto-update updated_at on any row change
CREATE OR REPLACE FUNCTION update_client_tiers_ts()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS client_tiers_ts ON client_tiers;
CREATE TRIGGER client_tiers_ts
    BEFORE UPDATE ON client_tiers
    FOR EACH ROW EXECUTE FUNCTION update_client_tiers_ts();

COMMENT ON TABLE client_tiers IS
    'Per-client rate limit tiers. client_id = SHA-256(raw_api_key).';
COMMENT ON COLUMN client_tiers.tier IS
    'free=10burst/1rps | pro=200burst/50rps | enterprise=2000burst/500rps';

-- ── Example rows (replace client_id with real SHA-256 hashes) ────────────────
-- INSERT INTO client_tiers (client_id, tier, notes)
-- VALUES
--   ('abc123...64chars', 'enterprise', 'Acme Corp pilot'),
--   ('def456...64chars', 'pro',        'Beta tester'),
--   ('ghi789...64chars', 'free',       'Sandbox account')
-- ON CONFLICT (client_id) DO UPDATE SET tier = EXCLUDED.tier;
