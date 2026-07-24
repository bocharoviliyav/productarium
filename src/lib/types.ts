/**
 * Shared domain types and helpers for the Productarium UI.
 *
 * These mirror the public Pydantic response shapes served by the FastAPI
 * backend (api/schemas.py: Product / Artifact / KnowledgeNode / UserOut /
 * ApiTokenOut / SettingOut) so the frontend stays in lock-step with the
 * API contract (contract J) without importing server code.
 */

/**
 * Artifact types exposed in the UI. `documentation` and `guides` are intentionally
 * omitted here — they are now authored as knowledge-tree pages instead of
 * artifacts. The backend still accepts/returns them for legacy data; the UI
 * falls back to a neutral label/icon when it encounters an unknown type.
 */
export type ArtifactType = "codebase" | "spec" | "links";

/** Any artifact type the backend may return, including legacy ones. */
export type ArtifactTypeAny = ArtifactType | "documentation" | "guides";

/** Subtype for `spec` artifacts (openapi/asyncapi). Optional elsewhere. */
export type ArtifactKind = "openapi" | "asyncapi" | string;

export type ArtifactSource = "manual" | "generated" | "api" | "mcp";

export interface ArtifactPage {
  id: string;
  title: string;
  content: string;
  filePaths?: string[];
  importance?: "high" | "medium" | "low";
  relatedPages?: string[];
}

/**
 * `Artifact.pages` is typed `Optional[Dict[str, Any]]` on the backend. After
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

export interface Artifact {
  id: string;
  name: string;
  type: ArtifactType;
  kind?: ArtifactKind | null;
  repo_url?: string | null;
  repo_type?: string | null;
  token?: string | null;
  content?: string | null;
  allure_url?: string | null;
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
  artifacts: Artifact[];
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

export type UserRole = "user" | "admin";
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
/* Tag / badge color mappings (desaturated pastels, minimalist-ui).    */
/* ------------------------------------------------------------------ */

export type TagTone =
  | "blue"
  | "green"
  | "yellow"
  | "red"
  | "neutral";

export const ARTIFACT_TYPE_META: Record<
  ArtifactType,
  { label: string; tone: TagTone; description: string }
> = {
  codebase: {
    label: "Codebase",
    tone: "blue",
    description: "Git repository documented from source.",
  },
  spec: {
    label: "Spec",
    tone: "green",
    description: "API / event contract (OpenAPI or AsyncAPI).",
  },
  links: {
    label: "Links",
    tone: "yellow",
    description: "Curated external links with descriptions.",
  },
};

/** Phosphor icon name hint per artifact type (resolved in components). */
export const ARTIFACT_TYPE_ICON: Record<ArtifactType, string> = {
  codebase: "GitBranch",
  spec: "FileText",
  links: "LinkSimple",
};

/**
 * Neutral fallback label/tone/icon for artifact types that the UI no longer
 * creates but the backend may still return (legacy `documentation`/`guides`).
 */
export const LEGACY_ARTIFACT_TYPE_META: {
  label: string;
  tone: TagTone;
  description: string;
  icon: string;
} = {
  label: "Archive",
  tone: "neutral",
  description: "Legacy artifact (now authored as a knowledge page).",
  icon: "Archive",
};

/** Resolve display metadata for an arbitrary artifact type string. */
export function artifactTypeMeta(
  type: string,
): { label: string; tone: TagTone; description: string } {
  if (type in ARTIFACT_TYPE_META) {
    return ARTIFACT_TYPE_META[type as ArtifactType];
  }
  return {
    label: LEGACY_ARTIFACT_TYPE_META.label,
    tone: LEGACY_ARTIFACT_TYPE_META.tone,
    description: LEGACY_ARTIFACT_TYPE_META.description,
  };
}

/** Resolve a Phosphor icon name for an arbitrary artifact type string. */
export function artifactTypeIcon(type: string): string {
  if (type in ARTIFACT_TYPE_ICON) {
    return ARTIFACT_TYPE_ICON[type as ArtifactType];
  }
  return LEGACY_ARTIFACT_TYPE_META.icon;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/**
 * A single curated link, as authored in the UI repeater and persisted as JSON
 * in `artifact.content` (matching the backend `_render_links_index` parser).
 */
export interface LinkItem {
  url: string;
  description?: string;
}

/**
 * Parse a `links` artifact's `content` into a list of link items.
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
 * an artifact's git fields. The backend RAG retriever is repo-keyed, so Q&A is
 * only meaningful for codebase artifacts that carry a repo_url. For non-git
 * artifacts this returns null and the UI shows a scoped note instead.
 */
export function artifactToRepoInfo(
  artifact: Artifact,
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
  if (artifact.type !== "codebase" || !artifact.repo_url) {
    return null;
  }
  let owner = "";
  let repo = "";
  try {
    const url = new URL(artifact.repo_url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length >= 2) {
      owner = parts[parts.length - 2];
      repo = (parts[parts.length - 1] || "").replace(/\.git$/, "");
    }
  } catch {
    const parts = artifact.repo_url.split("/").filter(Boolean);
    if (parts.length >= 2) {
      owner = parts[parts.length - 2];
      repo = (parts[parts.length - 1] || "").replace(/\.git$/, "");
    }
  }
  const type =
    artifact.repo_type ||
    (artifact.repo_url.includes("gitlab") ? "gitlab" : "github");
  return {
    owner,
    repo,
    type,
    token: artifact.token ?? null,
    localPath: null,
    repoUrl: artifact.repo_url,
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
export function generateId(prefix: "prod" | "art" | "node"): string {
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
