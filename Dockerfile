ARG CUSTOM_CERT_DIR="certs"

FROM node:22-alpine3.22 AS node_base

RUN npm install -g bun@1.4.2

FROM node_base AS node_deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile || (rm -rf ~/.bun/install/cache && bun install --frozen-lockfile)

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

FROM python:3.12-slim AS py312

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
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @bytebase/dbhub@1.2.3
RUN python -m pip install --no-cache-dir uv==0.8.14

COPY --from=py312 /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=py312 /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=py312 /usr/local/lib/libpython3.12.so.1.0 /usr/local/lib/libpython3.12.so.1.0

# The pinned checkout is MCP v1 code (``mcp.server.fastmcp``); upstream leaves
# ``mcp`` unpinned and v2 removed that module, so the pin is mandatory.
RUN ldconfig && python3.12 --version \
    && mkdir -p /opt/mcp \
    && uv venv /opt/mcp/oracle --python /usr/local/bin/python3.12 \
    && git clone https://github.com/danielmeppiel/oracle-mcp-server /opt/mcp/oracle/app \
    && git -C /opt/mcp/oracle/app checkout 37ce2ead4e8caa274eb9442b44aff7f7a59573dd \
    && uv pip install --python /opt/mcp/oracle/bin/python --no-cache -e /opt/mcp/oracle/app "mcp<2" \
    && rm -rf /opt/mcp/oracle/app/.git

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

COPY refs/ ./refs/

COPY tiktoken_cache/ /opt/tiktoken_cache/

# Copy Node app
COPY --from=node_builder /app/public ./public
COPY --from=node_builder /app/.next/standalone ./
COPY --from=node_builder /app/.next/static ./.next/static
# Runtime mermaid bundle for the headless diagram validator
# (api/_mermaid_validate.mjs). The .next/standalone output does NOT include
# it — Next bundles mermaid into static chunks — and the validator imports
# the package directly from node_modules. The ESM bundle is self-contained
# (relative chunks only, no external deps), so this one COPY is enough.
COPY --from=node_deps /app/node_modules/mermaid ./node_modules/mermaid

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
# loads with no network at first use.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

# Create empty .env file (will be overridden if one exists at runtime)
RUN touch .env

# Command to run the application
CMD ["/app/start.sh"]
