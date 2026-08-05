#!/usr/bin/env bash
# =============================================================================
# scripts/generate-key.sh — RouterFuel
#
# Generates a random client API key and its SHA-256 hash, formatted ready to
# paste into ROUTERFUEL_API_KEYS. Replaces manually running
# `echo -n "..." | sha256sum` from the README.
#
# Usage:
#   ./scripts/generate-key.sh "ClientName"
# =============================================================================

set -euo pipefail

CLIENT_NAME="${1:-Client}"

# 32 random bytes, hex-encoded, prefixed like the README's examples
RAW_KEY="rf_live_$(openssl rand -hex 24)"
HASH=$(echo -n "$RAW_KEY" | sha256sum | awk '{print $1}')

echo ""
echo "Give this key to your client (they use it as X-API-Key) — it will not be shown again:"
echo ""
echo "  $RAW_KEY"
echo ""
echo "Add this to ROUTERFUEL_API_KEYS (append with a comma if you already have entries):"
echo ""
echo "  ${HASH}:${CLIENT_NAME}"
echo ""
