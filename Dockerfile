ARG CUSTOM_CERT_DIR="certs"

# Deno version for fast-rlm (fast-rlm requires Deno 2+). Pinned for
# reproducibility; bump as needed. We pull the official binary from
# denoland/deno:bin-<ver> (a scratch-like image containing ONLY the deno
# binary at /deno) instead of the flaky `curl https://deno.land/install.sh`
# download, which was unreliable behind corporate proxies / flaky networks.
# The binary is glibc-linked; the final stage is python:3.11-slim
# (Debian/glibc), so there is no musl/glibc mismatch.
ARG DENO_VERSION=2.9.2
FROM denoland/deno:bin-${DENO_VERSION} AS deno_bin

FROM node:20-alpine3.22 AS node_base

# Install bun for frontend dependency install + build (Phase B: yarn -> bun).
RUN npm install -g bun

FROM node_base AS node_deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

FROM node_base AS node_builder
WORKDIR /app
COPY --from=node_deps /app/node_modules ./node_modules
# Copy only necessary files for Next.js build
COPY package.json bun.lock next.config.ts tsconfig.json tailwind.config.js postcss.config.mjs ./
COPY src/ ./src/
COPY public/ ./public/
# Increase Node.js memory limit for build and disable telemetry
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV NEXT_TELEMETRY_DISABLED=1
RUN NODE_ENV=production bun run build

FROM python:3.11-slim AS py_deps
WORKDIR /api
COPY api/pyproject.toml .
COPY api/poetry.lock .
RUN python -m pip install poetry==2.0.1 --no-cache-dir && \
    poetry config virtualenvs.create true --local && \
    poetry config virtualenvs.in-project true --local && \
    poetry config virtualenvs.options.always-copy --local true && \
    POETRY_MAX_WORKERS=10 poetry install --no-interaction --no-ansi --only main && \
    poetry cache clear --all .

# Use Python 3.11 as final image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Node.js, npm, git, unzip, and curl
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    unzip \
    ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the official Deno binary from the deno_bin stage (see top of file).
# Replaces the previous `curl -fsSL https://deno.land/install.sh | sh` install,
# which was the source of the flaky remote download for fast-rlm.
COPY --from=deno_bin /deno /usr/local/bin/deno

# Update certificates if custom ones were provided and copied successfully
RUN if [ -n "${CUSTOM_CERT_DIR}" ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        if [ -d "${CUSTOM_CERT_DIR}" ]; then \
            cp -r ${CUSTOM_CERT_DIR}/* /usr/local/share/ca-certificates/ 2>/dev/null || true; \
            update-ca-certificates; \
            echo "Custom certificates installed successfully."; \
        else \
            echo "Warning: ${CUSTOM_CERT_DIR} not found. Skipping certificate installation."; \
        fi \
    fi

ENV PATH="/opt/venv/bin:$PATH"

# Copy Python dependencies
COPY --from=py_deps /api/.venv /opt/venv
COPY api/ ./api/
# Copy the externalized prompt bodies (refs/prompts/*.md) and reference docs.
# api/prompts.py loads these at import time via load_prompt_file(); without
# this copy, every WIKI_*_PROMPT constant is empty inside the container and
# ALL section prompts (overview/architecture/.../datamodel) are blank -- which
# silently breaks RLM (no instruction, just raw codebase) and the standard-LLM
# fallback (empty user message -> LM Studio "No user query found").
COPY refs/ ./refs/

# Pre-vendor tiktoken's cl100k_base BPE mergeable-ranks file (offline fix).
# adalflow's TextSplitter/Tokenizer call tiktoken.get_encoding("cl100k_base")
# at import time, which otherwise downloads cl100k_base.tiktoken from
# openaipublic.blob.core.windows.net -- unreachable offline in this container,
# crashing startup with ConnectionError. tiktoken's read_file_cached()
# (tiktoken/load.py) names the cache file by sha1(download_url) and serves it
# with no network when the file exists and its sha256 matches the expected
# hash. We vendor that verified file (sha256
# 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7) named
# 9b5ad71b2ce5302211f9c61530b329a4922fc6a4 (= sha1 of the blob URL).
# TIKTOKEN_CACHE_DIR below points tiktoken at this dir. Regenerating the file:
#   curl -fsSL -o tiktoken_cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 \
#     https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
# (then verify sha256 matches the hash above).
COPY tiktoken_cache/ /opt/tiktoken_cache/

# Pre-cache fast-rlm's Deno dependencies (jsr @std/yaml, npm openai/pyodide/...)
# at BUILD time. Without this, the first generate request triggers a runtime
# fetch of https://jsr.io/@std/yaml/... which can fail with a TLS error
# ("peer closed connection without sending TLS close_notify"), crashing RLM.
# The cache lands in /root/.deno, which matches the DENO_DIR the compose file
# mounts as the `deno_cache` named volume: on the volume's first mount Docker
# copies this pre-populated dir into it, so the deps persist across restarts.
# (If you already have a deno_cache volume from an older image, remove it --
# `docker compose down -v` or `docker volume rm <project>_deno_cache` -- so the
# fresh build-time cache is picked up.) Best-effort: if jsr.io/npm is
# unreachable during build, RLM retries at runtime and the standard-LLM
# fallback covers any failure.
ENV DENO_DIR=/root/.deno
RUN ENGINE_DIR="$(/opt/venv/bin/python -c 'from fast_rlm._runner import _find_engine_dir; print(_find_engine_dir())' 2>/dev/null)" \
    && if [ -n "$ENGINE_DIR" ] && [ -f "$ENGINE_DIR/src/subagents.ts" ]; then \
        cd "$ENGINE_DIR" \
        && deno cache src/subagents.ts \
        || echo "Warning: deno cache failed (non-fatal); RLM will fetch deps at runtime."; \
    else \
        echo "Warning: fast-rlm engine not found; skipping Deno dep pre-cache."; \
    fi

# Copy Node app
COPY --from=node_builder /app/public ./public
COPY --from=node_builder /app/.next/standalone ./
COPY --from=node_builder /app/.next/static ./.next/static

# Expose the port the app runs on
EXPOSE ${PORT:-8001} 3000

# Create a script to run both backend and frontend
RUN echo '#!/bin/bash\n\
# Load environment variables from .env file if it exists\n\
if [ -f .env ]; then\n\
  export $(grep -v "^#" .env | xargs -r)\n\
fi\n\
\n\
# Start the API server in the background with the configured port\n\
python -m api.main --port ${PORT:-8001} &\n\
PORT=3000 HOSTNAME=0.0.0.0 node server.js &\n\
wait -n\n\
exit $?' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV PORT=8001
ENV NODE_ENV=production
ENV SERVER_BASE_URL=http://localhost:${PORT:-8001}
# Point tiktoken at the vendored BPE cache (see COPY above) so cl100k_base
# loads with no network at adalflow import time.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

# Create empty .env file (will be overridden if one exists at runtime)
RUN touch .env

# Command to run the application
CMD ["/app/start.sh"]
