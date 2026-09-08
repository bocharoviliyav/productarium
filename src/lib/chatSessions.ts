/**
 * Chat session history client (wave B — persistent expert chat).
 *
 * Backend contract (owned by the wave-b backend agent; shapes normalized
 * defensively because the exact response envelope is still landing):
 * - `GET  /api/products/{productId}/chat/sessions` — list sessions.
 * - `GET  /api/products/{productId}/chat/sessions/{sessionId}/messages` —
 *   ordered chat history for one session.
 *
 * Both may arrive as a bare JSON array or wrapped in `{sessions: [...]}` /
 * `{items: [...]}` / `{messages: [...]}`; ids and dates may be `id`/`session_id`
 * and `created_at`/`createdAt`. Every reader tolerates 404 (feature not yet
 * deployed) by resolving to an empty list so the chat degrades gracefully.
 */

/** A persisted chat session (list view). */
export interface ChatSession {
  id: string;
  /** Optional human title shown in the session strip. */
  title: string | null;
  createdAt: string | null;
  updatedAt: string | null;
}

/** A persisted chat message inside a session. */
export interface ChatMessage {
  /** Backend also persists tool rows (``role='tool'``) with a short summary. */
  role: "user" | "assistant" | "tool";
  content: string;
  createdAt: string | null;
  /** Tool name for role='tool' rows (null otherwise). */
  toolName: string | null;
  /** Parsed tool args for role='tool' rows (null when absent/unparseable). */
  toolArgs: Record<string, unknown> | null;
}

/** Error carrying the HTTP status so callers can react to 401/404. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function str(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

/** Extract an array of unknown records from a JSON envelope or array. */
function unwrapList(payload: unknown, keys: readonly string[]): Record<string, unknown>[] {
  if (Array.isArray(payload)) return payload.filter(isRecord);
  if (isRecord(payload)) {
    for (const key of keys) {
      const field = payload[key];
      if (Array.isArray(field)) return field.filter(isRecord);
    }
  }
  return [];
}

function normalizeDate(value: unknown): string | null {
  if (typeof value !== "string" || !value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toISOString();
}

/** Normalize one session record from any of the tolerated field spellings. */
export function normalizeSession(raw: Record<string, unknown>): ChatSession {
  return {
    id: str(raw.id) ?? str(raw.session_id) ?? "",
    title: str(raw.title) ?? str(raw.name) ?? str(raw.summary),
    createdAt: normalizeDate(raw.created_at ?? raw.createdAt),
    updatedAt: normalizeDate(raw.updated_at ?? raw.updatedAt),
  };
}

/** Parse a stored tool_args JSON string into a record (null when invalid). */
function parseToolArgs(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "string" || !value) return null;
  try {
    const parsed: unknown = JSON.parse(value);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    /* fall through */
  }
  return null;
}

/** Normalize one message record into the UI view model. */
export function normalizeMessage(raw: Record<string, unknown>): ChatMessage | null {
  const content = str(raw.content);
  if (!content) return null;
  const rawRole = str(raw.role) ?? str(sender(raw));
  const role =
    rawRole === "tool"
      ? "tool"
      : rawRole === "assistant" || rawRole === "ai"
        ? "assistant"
        : "user";
  return {
    role,
    content,
    createdAt: normalizeDate(raw.created_at ?? raw.createdAt),
    toolName: str(raw.tool_name) ?? str(raw.toolName),
    toolArgs: parseToolArgs(raw.tool_args ?? raw.toolArgs),
  };
}

function sender(raw: Record<string, unknown>): string | null {
  // Tolerate {"sender": {"type": "human"}}-style envelopes.
  if (isRecord(raw.sender)) return str(raw.sender.type);
  return null;
}

/**
 * Load the session list for a product.
 *
 * Resolves to `[]` on 404 (sessions endpoint not deployed yet) so the chat
 * UI stays usable. Throws `ApiError` for other failures (incl. 401).
 */
export async function fetchChatSessions(productId: string): Promise<ChatSession[]> {
  const res = await fetch(
    `/api/products/${encodeURIComponent(productId)}/chat/sessions`,
    { credentials: "include", cache: "no-store" },
  );
  if (res.status === 404) return [];
  if (res.status === 401) throw new ApiError(401, "Not authenticated");
  if (!res.ok) throw new ApiError(res.status, `Failed to load sessions (${res.status})`);
  const payload: unknown = await res.json().catch(() => null);
  return unwrapList(payload, ["sessions", "items", "data"])
    .map(normalizeSession)
    .filter((s) => s.id);
}

/**
 * Load the message history of one session.
 *
 * Resolves to `[]` on 404. Messages are returned in the order the backend
 * sends them (oldest first); a newest-first response is detected by
 * timestamps and reversed for display.
 */
export async function fetchChatMessages(
  productId: string,
  sessionId: string,
): Promise<ChatMessage[]> {
  const res = await fetch(
    `/api/products/${encodeURIComponent(productId)}/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
    { credentials: "include", cache: "no-store" },
  );
  if (res.status === 404) return [];
  if (res.status === 401) throw new ApiError(401, "Not authenticated");
  if (!res.ok) {
    throw new ApiError(res.status, `Failed to load messages (${res.status})`);
  }
  const payload: unknown = await res.json().catch(() => null);
  const messages = unwrapList(payload, ["messages", "items", "data", "history"])
    .map(normalizeMessage)
    .filter((m): m is ChatMessage => m !== null);
  return maybeReverse(messages);
}

/** Reverse newest-first lists so history renders oldest → newest. */
function maybeReverse(messages: ChatMessage[]): ChatMessage[] {
  const stamps = messages
    .map((m) => (m.createdAt ? Date.parse(m.createdAt) : Number.NaN))
    .filter((t) => !Number.isNaN(t));
  if (stamps.length < 2) return messages;
  const ascending = stamps.every((t, i) => i === 0 || t >= stamps[i - 1]);
  if (ascending) return messages;
  return messages.slice().reverse();
}
