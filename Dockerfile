# =============================================================================
# Dockerfile — RouterFuel
#
# Two-stage build: compile in a full Rust image, run in a slim Debian image.
#
# Honest caveats before you build this:
#   1. `ort` (ONNX Runtime bindings, used for local semantic-cache embeddings)
#      downloads prebuilt onnxruntime binaries during `cargo build` via its
#      "download-binaries" feature — this build needs network access, and
#      the runtime .so it produces has to be found by the final binary via
#      LD_LIBRARY_PATH (handled below). This is the single most likely thing
#      to need a fix on your first build — if `docker compose up` starts but
#      logs a warning that the embedding model couldn't load, that's this,
#      and the gateway will still run fine WITHOUT semantic caching (see
#      src/embedder.rs / main.rs — that failure path is already handled
#      gracefully in your own code, it just disables one feature, not the
#      whole server).
#   2. `tokenizers` is built with the "onig" feature, which compiles the C
#      Oniguruma library from source — needs a C compiler in the build
#      image (installed below).
#   3. sqlx uses `runtime-tokio-native-tls`, so the build needs OpenSSL dev
#      headers, and the runtime image needs libssl at runtime.
#   4. Good news: cost_tracker.rs/admin.rs/etc. all use sqlx::query() (the
#      runtime-checked form), not the sqlx::query!() compile-time macro — so
#      this build does NOT need a live database connection or
#      DATABASE_URL/SQLX_OFFLINE at build time. One less thing to fight.
# =============================================================================

# ---- Build stage ----
# Pinned to the floating "1-bookworm" tag (always latest stable Rust 1.x) —
# not a fixed version like "1.82" — because Cargo.lock was generated with
# whatever Rust version is on the host machine, and a fixed older tag here
# can fall behind what the lockfile actually needs (this bit us once
# already: home v0.5.12 requires the `edition2024` Cargo feature, which
# needs Cargo 1.85+; the original 1.82 pin didn't have it).
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    clang \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependency compilation separately from source changes: build a
# throwaway main.rs first so `cargo build` compiles every dependency (the
# slow part, especially `ort`) into a layer that's reused as long as
# Cargo.toml/Cargo.lock don't change, even when src/ changes on every edit.
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src \
    && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

# Now the real source. migrations/ and static/ are both pulled in at COMPILE
# time, not runtime — sqlx::migrate!("./migrations") embeds the SQL files
# into the binary, and main.rs's dashboard_handler uses
# include_str!("../static/dashboard.html") — so both must be present here,
# but neither needs to exist in the final runtime image.
COPY src ./src
COPY migrations ./migrations
COPY static ./static

RUN cargo build --release

# Collect any ONNX Runtime shared libraries into a directory that's
# guaranteed to exist (even if empty) — a direct wildcard COPY across
# stages fails the whole build if it matches zero files, which would be a
# much worse failure mode than "semantic caching is quietly disabled."
RUN mkdir -p /app/onnxlibs \
    && cp /app/target/release/*.so* /app/onnxlibs/ 2>/dev/null || true

# ---- Runtime stage ----
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/routerfuel ./routerfuel

# Directory is guaranteed to exist from the builder stage (see the `mkdir -p`
# there), even if it ended up empty — so this COPY can't fail the build the
# way a direct wildcard cross-stage COPY can. If it's empty, the gateway
# still starts fine; semantic caching just stays disabled until the library
# path is sorted out. See caveat #1 at the top of this file.
COPY --from=builder /app/onnxlibs ./onnxlibs
ENV LD_LIBRARY_PATH=/app/onnxlibs:/app

EXPOSE 3000

CMD ["./routerfuel"]
