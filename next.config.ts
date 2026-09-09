import type { NextConfig } from "next";

const TARGET_SERVER_BASE_URL =
  process.env.SERVER_BASE_URL || "http://localhost:8001";

// Timeout (ms) for the /api/* rewrite proxy. The Next.js default (30 s) used
// to kill long-running expert SSE streams mid-answer (issue #9); default here
// is 1 h to match the backend expert_stream budget (EXPERT_STREAM_TIMEOUT_
// SECONDS). Override with NEXT_PROXY_TIMEOUT_MS (read at server start).
const PROXY_TIMEOUT_MS = Number.parseInt(
  process.env.NEXT_PROXY_TIMEOUT_MS || "3600000",
  10,
);

const nextConfig: NextConfig = {
  /* config options here */
  output: "standalone",
  // Optimize build for Docker
  experimental: {
    optimizePackageImports: ["@mermaid-js/mermaid", "react-syntax-highlighter"],
    ...(Number.isFinite(PROXY_TIMEOUT_MS) && PROXY_TIMEOUT_MS > 0
      ? { proxyTimeout: PROXY_TIMEOUT_MS }
      : {}),
  },
  // Reduce memory usage during build
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    // Optimize bundle size
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: "all",
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: "vendors",
            chunks: "all",
          },
        },
      },
    };
    return config;
  },
  async rewrites() {
    return [
      // Product/Artifact CRUD endpoints (product-centric routing), including
      // the expert chat + chat sessions. These have no Next.js route handler;
      // proxy straight to the FastAPI backend.
      {
        source: "/api/products",
        destination: `${TARGET_SERVER_BASE_URL}/api/products`,
      },
      {
        source: "/api/products/:path*",
        destination: `${TARGET_SERVER_BASE_URL}/api/products/:path*`,
      },
      // Database preset catalog (GET /api/db-presets) for the add-database
      // dialog: top-level (not under /api/products), so it needs its own rule.
      {
        source: "/api/db-presets",
        destination: `${TARGET_SERVER_BASE_URL}/api/db-presets`,
      },
      // Auth (contract J): local login / me / logout / keycloak. Generic
      // catch-all after the legacy specific routes below.
      {
        source: "/api/auth/status",
        destination: `${TARGET_SERVER_BASE_URL}/auth/status`,
      },
      {
        source: "/api/auth/validate",
        destination: `${TARGET_SERVER_BASE_URL}/auth/validate`,
      },
      {
        source: "/api/auth/:path*",
        destination: `${TARGET_SERVER_BASE_URL}/api/auth/:path*`,
      },
      // Admin panel (contract J): /api/admin/{group} + /api/admin/{group}/test.
      {
        source: "/api/admin/:path*",
        destination: `${TARGET_SERVER_BASE_URL}/api/admin/:path*`,
      },
      // API tokens (admin creates/revokes; listed under /api/admin/apitokens
      // above, but also expose the no-suffix create endpoint).
      {
        source: "/api/admin",
        destination: `${TARGET_SERVER_BASE_URL}/api/admin`,
      },
      {
        source: "/api/lang/config",
        destination: `${TARGET_SERVER_BASE_URL}/lang/config`,
      },
    ];
  },
};

export default nextConfig;
