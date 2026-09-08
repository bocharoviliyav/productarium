/**
 * Shared domain types and helpers for the Productarium UI.
 *
 * These mirror the public Pydantic response shapes served by the FastAPI
 * backend (api/schemas.py: Product / Codebase / Spec / Links / KnowledgeNode /
 * UserOut / ApiTokenOut / SettingOut) so the frontend stays in lock-step with
 * the API contract (contract J) without importing server code.
 */

export type ArtifactSource = "manual" | "generated" | "api" | "mcp";

/* ------------------------------------------------------------------ */
/* Pages (codebase wiki page tree)                                      */
/* ------------------------------------------------------------------ */

export interface ArtifactPage {
  id: string;
  title: string;
  content: string;
  /**
   * Parent page id for generated subpages (docgen units model: children of
   * the functional/technical/datamodel sections carry the parent's page id).
   * Legacy pages have no parent and render flat — the viewer must tolerate
   * orphaned parents/children without breaking.
   */
  parent?: string;
  filePaths?: string[];
  importance?: "high" | "medium" | "low";
  relatedPages?: string[];
  /**
   * Additive provenance block persisted by the verification pipeline
   * (api/docgen/verification.py). Unknown to legacy pages — the viewer must
   * tolerate its presence without breaking.
   */
  provenance?: Record<string, unknown>;
}

/**
 * `Codebase.pages` is typed `Optional[Dict[str, Any]]` on the backend. After
 * the doc-gen agent landed, the generate endpoint persists `pages` as a dict
 * keyed by page id: `{ [page_id]: WikiPage }` (WikiPage =
 * { id, title, content, filePaths, importance, relatedPages }). The viewer
 * must also tolerate the array form and the `{ pages: WikiPage[] }` wrapper,
 * plus null (generate may have only written generated_docs).
 */
export type ArtifactPages =
  | Record<string, ArtifactPage>
  | ArtifactPage[]
  | { pages?: ArtifactPage[] }
  | null
  | undefined;

/* ------------------------------------------------------------------ */
/* Codebase / Spec / Links / Product                                    */
/* ------------------------------------------------------------------ */

export interface Codebase {
  id: string;
  name: string;
  repo_url?: string | null;
  repo_type?: string | null;
  /**
   * Write-only field: the backend accepts a token on create/update but never
   * returns it (Pydantic `exclude`). Responses carry `has_token` instead —
   * the token itself is resolved server-side from encrypted storage.
   */
  token?: string | null;
  /** True when the backend has a stored (encrypted) access token. */
  has_token?: boolean;
  generated_docs?: string | null;
  pages?: ArtifactPages;
  verified?: boolean;
  verified_by?: string | null;
  verified_at?: string | null;
  source?: ArtifactSource;
}

export type SpecKind = "openapi" | "asyncapi";

export interface Spec {
  id: string;
  name: string;
  kind: SpecKind;
  content?: string | null;
  verified?: boolean;
  verified_by?: string | null;
  verified_at?: string | null;
  source?: ArtifactSource;
}

export interface Links {
  id: string;
  name: string;
  content?: string | null; // JSON array of {url, description}
  verified?: boolean;
  verified_by?: string | null;
  verified_at?: string | null;
  source?: ArtifactSource;
}

/**
 * A reverse-engineered database attached to a product (wave E).
 *
 * The raw DSN never leaves the server: reads return `dsn_masked` only.
 * `mcp_server_id` references the admin MCP registry entry whose tools are
 * used for the reverse-engineering flow; the backend may additionally serve
 * `mcp_server_name` for display (tolerated, not required).
 */
export interface Database {
  id: string;
  name: string;
  dsn_masked?: string | null;
  mcp_server_id?: string | null;
  mcp_server_name?: string | null;
  generated_docs?: string | null;
  pages?: ArtifactPages;
  verified?: boolean;
  verified_by?: string | null;
  verified_at?: string | null;
  source?: ArtifactSource;
}

export interface Product {
  id: string;
  name: string;
  description: string;
  /** AI-generated product summary (item 4). */
  summary?: string | null;
  /** Owner user id (FK users.id). */
  owner_id?: string | null;
  codebases: Codebase[];
  specs: Spec[];
  links: Links[];
  /**
   * Reverse-engineered databases (wave E). Optional so the UI keeps working
   * against backends that do not serve the field yet — always read via
   * `product.databases ?? []`.
   */
  databases?: Database[];
}

/**
 * Light product row served by GET /api/products (P1-16): children appear
 * only as SQL-counted totals — no child payloads. The endpoint keeps the
 * bare JSON-array response shape (no envelope): pagination goes via
 * `limit`/`offset` query params and the filtered total rides in the
 * `X-Total-Count` response header. The full object is served only by
 * GET /api/products/{id}. Counter fields are optional so a full `Product`
 * (e.g. the POST create response prepended to a list) is assignable to
 * this type — read counters via `?? child?.length ?? 0` fallbacks.
 */
export interface ProductListItem {
  id: string;
  name: string;
  description: string;
  summary?: string | null;
  owner_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  codebases_count?: number;
  specs_count?: number;
  links_count?: number;
  databases_count?: number;
  verified_codebases?: number;
  verified_specs?: number;
  verified_links?: number;
  verified_databases?: number;
}

/* ------------------------------------------------------------------ */
/* Knowledge tree (Confluence-like, item 2)                            */
/* ------------------------------------------------------------------ */

export type KnowledgeNodeType = "page" | "folder" | "branch";

export interface KnowledgeNode {
  id: string;
  product_id: string;
  parent_id?: string | null;
  title: string;
  slug: string;
  content_md?: string | null;
  node_type: KnowledgeNodeType;
  artifact_id?: string | null;
  source?: ArtifactSource;
  verified?: boolean;
  verified_by?: string | null;
  verified_at?: string | null;
  created_by?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  /** Nested children when the tree endpoint returns a nested structure. */
  children?: KnowledgeNode[];
}

/* ------------------------------------------------------------------ */
/* Auth + admin (contract J)                                           */
/* ------------------------------------------------------------------ */

export type UserRole = "user" | "admin" | "manager" | "viewer_global";
export type AuthProvider = "local" | "keycloak";

export interface User {
  id: string;
  username: string;
  email?: string | null;
  role: UserRole;
  provider: AuthProvider;
  created_at?: string | null;
  /** True when the user must change their password on next login (temp password). */
  must_change_password?: boolean;
}

/** First-run setup probe (GET /api/auth/setup-status). */
export interface SetupStatus {
  setup_required: boolean;
  auth_provider: string;
}

/** Result of admin user creation (temp password + reset token shown once). */
export interface UserCreateResult {
  user: User;
  temp_password?: string | null;
  reset_token?: string | null;
}

export interface ApiToken {
  id: string;
  name: string;
  created_at?: string | null;
  last_used_at?: string | null;
  /** Raw token returned only once at creation time. */
  token?: string | null;
}

export interface SettingOut {
  key: string;
  value?: string | null;
  encrypted: boolean;
}

/* ------------------------------------------------------------------ */
/* MCP servers (admin registry + per-product bindings, wave C)         */
/* ------------------------------------------------------------------ */

export type McpTransport = "http" | "stdio";
export type McpServerStatus = "ok" | "error" | "unknown";

/**
 * Registry entry served by the admin MCP registry
 * (GET/POST/PUT /api/admin/mcp/servers, POST .../test). Header and env
 * values come back masked — secrets never leave the server.
 */
export interface McpServer {
  id: string;
  name: string;
  transport: McpTransport;
  url?: string | null;
  command?: string | null;
  args?: string[] | null;
  headers_masked?: Record<string, string> | null;
  env_masked?: Record<string, string> | null;
  enabled: boolean;
  status: McpServerStatus;
  status_checked_at?: string | null;
  status_error?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

/** A tool discovered on an MCP server (name + optional description). */
export interface McpToolInfo {
  name: string;
  description?: string | null;
}

/** Result of POST /api/admin/mcp/servers/{id}/test (health + discovery). */
export interface McpTestResult {
  ok: boolean;
  detail?: string | null;
  tools: McpToolInfo[];
}

/**
 * Per-product binding served by /api/products/{product_id}/mcp.
 * `allowed_tools: null` means every tool of the bound server is exposed to
 * the expert agent (no allowlist filter).
 */
export interface McpServerBinding {
  id: string;
  mcp_server_id: string;
  name: string;
  transport: McpTransport;
  enabled: boolean;
  allowed_tools: string[] | null;
  status: McpServerStatus;
}

/* ------------------------------------------------------------------ */
/* Entity kinds (codebase / spec / links)                               */
/* ------------------------------------------------------------------ */

export type EntityKind = "codebase" | "spec" | "links" | "database";

/**
 * Backend URL segment for an entity kind.
 *
 * The FastAPI routers (``api/routers/products.py``, ``docgen.py``) register
 * these sub-resources under the PLURAL segments (``codebases`` / ``specs`` /
 * ``links`` / ``databases``). The UI stores the SINGULAR kind
 * (``codebase`` / ``spec`` / ``links`` / ``database``) in form state, so every
 * API URL must go through this helper — interpolating the raw kind directly
 * produces 404s.
 */
export function entityPath(kind: EntityKind): string {
  switch (kind) {
    case "codebase":
      return "codebases";
    case "spec":
      return "specs";
    case "links":
      return "links";
    case "database":
      return "databases";
  }
}

/* ------------------------------------------------------------------ */
/* Tag / badge color mappings (desaturated pastels, minimalist-ui).    */
/* ------------------------------------------------------------------ */

export type TagTone =
  | "blue"
  | "green"
  | "yellow"
  | "red"
  | "neutral";

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/**
 * A single curated link, as authored in the UI repeater and persisted as JSON
 * in `links.content` (matching the backend `_render_links_index` parser).
 */
export interface LinkItem {
  url: string;
  description?: string;
}

/**
 * Parse a `links` entity's `content` into a list of link items.
 *
 * The backend accepts several shapes: a JSON array of `{url, description?,
 * title?}` objects, a `{ links: [...] }` wrapper, a single link object, or
 * free-form Markdown (returned as a single item with the raw text as url so
 * the editor can still show/edit it). Empty content yields `[]`.
 */
export function parseLinksContent(content: string | null | undefined): LinkItem[] {
  const text = (content ?? "").trim();
  if (!text) return [];
  try {
    const loaded = JSON.parse(text);
    const norm = (obj: unknown): LinkItem | null => {
      if (!obj || typeof obj !== "object") return null;
      const o = obj as Record<string, unknown>;
      const url = String(o.url ?? o.link ?? "").trim();
      const description = String(o.description ?? o.desc ?? o.title ?? "").trim();
      if (!url && !description) return null;
      return { url, description };
    };
    if (Array.isArray(loaded)) {
      return loaded.map(norm).filter((x): x is LinkItem => x !== null);
    }
    if (loaded && typeof loaded === "object") {
      const linksField = (loaded as Record<string, unknown>).links;
      if (Array.isArray(linksField)) {
        return linksField.map(norm).filter((x): x is LinkItem => x !== null);
      }
      const single = norm(loaded);
      if (single) return [single];
    }
  } catch {
    /* fall through to markdown handling */
  }
  // Free-form Markdown / legacy "url | description" lines.
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [url, ...rest] = line.split("|");
      const description = rest.join("|").trim();
      return { url: (url ?? "").trim(), description };
    })
    .filter((it) => it.url || it.description);
}

/** Serialize a list of link items back to the JSON string form for storage. */
export function serializeLinksContent(items: LinkItem[]): string {
  return JSON.stringify(
    items
      .filter((it) => it.url.trim() || (it.description ?? "").trim())
      .map((it) => ({ url: it.url.trim(), description: (it.description ?? "").trim() })),
  );
}

/**
 * Build a RepoInfo (consumed by the existing Ask / WebSocket chat path) from
 * a codebase's git fields. The backend RAG retriever is repo-keyed, so Q&A is
 * only meaningful for codebases that carry a repo_url. Returns null otherwise
 * and the UI shows a scoped note instead.
 */
export function codebaseToRepoInfo(
  codebase: Codebase,
):
  | {
      owner: string;
      repo: string;
      type: string;
      token: string | null;
      localPath: string | null;
      repoUrl: string | null;
    }
  | null {
  if (!codebase.repo_url) {
    return null;
  }
  let owner = "";
  let repo = "";
  try {
    const url = new URL(codebase.repo_url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length >= 2) {
      owner = parts[parts.length - 2];
      repo = (parts[parts.length - 1] || "").replace(/\.git$/, "");
    }
  } catch {
    const parts = codebase.repo_url.split("/").filter(Boolean);
    if (parts.length >= 2) {
      owner = parts[parts.length - 2];
      repo = (parts[parts.length - 1] || "").replace(/\.git$/, "");
    }
  }
  const type =
    codebase.repo_type ||
    (codebase.repo_url.includes("gitlab") ? "gitlab" : "github");
  return {
    owner,
    repo,
    type,
    // The API no longer returns tokens (write-only): the backend resolves the
    // stored encrypted token itself when cloning/pulling.
    token: null,
    localPath: null,
    repoUrl: codebase.repo_url,
  };
}

/** Normalize the free-form `pages` field into a flat list of pages.
 *
 * Handles three shapes the backend may persist:
 *  - Record<page_id, ArtifactPage>  (current generate-endpoint output)
 *  - ArtifactPage[]                  (legacy / ad-hoc)
 *  - { pages: ArtifactPage[] }        (wiki-structure wrapper)
 */
export function normalizePages(pages: ArtifactPages): ArtifactPage[] {
  if (!pages) return [];
  if (Array.isArray(pages)) return pages as ArtifactPage[];
  if (Array.isArray((pages as { pages?: ArtifactPage[] }).pages)) {
    return (pages as { pages: ArtifactPage[] }).pages;
  }
  // Dict keyed by page id -> page object.
  if (typeof pages === "object") {
    const values = Object.values(pages as Record<string, unknown>);
    if (values.every((v) => v && typeof v === "object" && "id" in (v as object))) {
      return values as ArtifactPage[];
    }
  }
  return [];
}

/** A short, stable id for new client-created entities before the server roundtrip. */
export function generateId(
  prefix: "prod" | "cb" | "spec" | "links" | "db" | "node",
): string {
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now().toString(36)}${rand}`;
}

/** Derive a URL-safe slug from a title (mirrors the backend slug derivation). */
export function slugify(title: string): string {
  return title
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .slice(0, 64);
}

/** Build a nested tree from a flat list of knowledge nodes (by parent_id). */
export function buildKnowledgeTree(nodes: KnowledgeNode[]): KnowledgeNode[] {
  const byParent = new Map<string | null, KnowledgeNode[]>();
  for (const n of nodes) {
    const key = n.parent_id ?? null;
    const list = byParent.get(key) ?? [];
    list.push(n);
    byParent.set(key, list);
  }
  const attach = (parent: string | null): KnowledgeNode[] =>
    (byParent.get(parent) ?? []).map((n) => ({
      ...n,
      children: attach(n.id),
    }));
  const sortRec = (list: KnowledgeNode[]): KnowledgeNode[] =>
    list
      .slice()
      .sort((a, b) => a.title.localeCompare(b.title))
      .map((n) => ({ ...n, children: sortRec(n.children ?? []) }));
  return sortRec(attach(null));
}
