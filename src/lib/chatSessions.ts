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

/** Attachment chip metadata riding a user message. */
export interface ChatAttachmentMeta {
  id: string;
  filename: string;
  sizeBytes: number;
}

/** Upload result (POST …/ask/attachments). */
export interface UploadedAttachment extends ChatAttachmentMeta {
  contentChars: number;
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
  /** Attachment chips on user rows (empty otherwise). */
  attachments: ChatAttachmentMeta[];
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

/** Normalize attachment chip records (tolerates field spellings). */
function normalizeAttachments(value: unknown): ChatAttachmentMeta[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((raw) => ({
      id: str(raw.id) ?? "",
      filename: str(raw.filename) ?? str(raw.name) ?? "",
      sizeBytes:
        typeof raw.size_bytes === "number" || typeof raw.sizeBytes === "number"
          ? ((raw.size_bytes ?? raw.sizeBytes) as number)
          : 0,
    }))
    .filter((a) => a.id);
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
    attachments: role === "user" ? normalizeAttachments(raw.attachments) : [],
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
  const ascending = stamps.every((t, i) => t >= stamps[i - 1]);
  if (ascending) return messages;
  return messages.slice().reverse();
}

/* ------------------------------------------------------------------ */
/* Detached turns (issue #9: generation survives disconnects)         */
/* ------------------------------------------------------------------ */

/**
 * A RUNNING ask turn of a session (GET …/active-turn). Null when the session
 * has nothing generating. The UI re-attaches via the SSE stream endpoint
 * using `turnId` to catch the buffered answer + live tail.
 */
export interface ActiveTurnInfo {
  turnId: string;
  sessionId: string | null;
  status: string;
  query: string | null;
  startedAt: string | null;
  finishedAt: string | null;
}

/** Normalize the active-turn descriptor (tolerates field spellings). */
function normalizeActiveTurn(raw: Record<string, unknown>): ActiveTurnInfo | null {
  const turnId = str(raw.turn_id) ?? str(raw.turnId) ?? str(raw.id);
  if (!turnId) return null;
  return {
    turnId,
    sessionId: str(raw.session_id) ?? str(raw.sessionId),
    status: str(raw.status) ?? "running",
    query: str(raw.query),
    startedAt: normalizeDate(raw.started_at ?? raw.startedAt),
    finishedAt: normalizeDate(raw.finished_at ?? raw.finishedAt),
  };
}

/**
 * Probe a session for a RUNNING turn (the re-attach check). Resolves to null
 * on 404 / `null` body / parse issues — no running generation.
 */
export async function fetchActiveTurn(
  productId: string,
  sessionId: string,
): Promise<ActiveTurnInfo | null> {
  let res: Response;
  try {
    res = await fetch(
      `/api/products/${encodeURIComponent(productId)}/chat/sessions/${encodeURIComponent(sessionId)}/active-turn`,
      { credentials: "include", cache: "no-store" },
    );
  } catch {
    return null;
  }
  if (!res.ok) return null;
  const payload: unknown = await res.json().catch(() => null);
  if (!isRecord(payload)) return null;
  return normalizeActiveTurn(payload);
}

/**
 * Stop a running turn (the Stop button). The partial answer is persisted
 * server-side; resolves to the final turn status ("cancelled" on success).
 */
export async function cancelExpertTurn(
  productId: string,
  turnId: string,
): Promise<string> {
  const res = await fetch(
    `/api/products/${encodeURIComponent(productId)}/ask/${encodeURIComponent(turnId)}/cancel`,
    { method: "POST", credentials: "include" },
  );
  if (res.status === 404) return "unknown";
  if (!res.ok) throw new ApiError(res.status, `Failed to cancel turn (${res.status})`);
  const payload: unknown = await res.json().catch(() => null);
  return isRecord(payload) && typeof payload.status === "string"
    ? payload.status
    : "cancelled";
}

/**
 * Upload chat attachments (multipart) — returns metadata chips whose ids ride
 * the next ask. Renditions are conversation-context only, never indexed.
 */
export async function uploadChatAttachments(
  productId: string,
  files: File[],
): Promise<UploadedAttachment[]> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch(
    `/api/products/${encodeURIComponent(productId)}/ask/attachments`,
    { method: "POST", body: form, credentials: "include" },
  );
  if (res.status === 401) throw new ApiError(401, "Not authenticated");
  if (!res.ok) {
    const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
    throw new ApiError(
      res.status,
      detail?.detail ?? `Attachment upload failed (${res.status})`,
    );
  }
  const payload: unknown = await res.json().catch(() => null);
  return unwrapList(payload, ["attachments", "items", "data"])
    .map((raw) => ({
      id: str(raw.id) ?? "",
      filename: str(raw.filename) ?? "",
      sizeBytes: typeof raw.size_bytes === "number" ? raw.size_bytes : 0,
      contentChars: typeof raw.content_chars === "number" ? raw.content_chars : 0,
    }))
    .filter((a) => a.id);
}

/** Direct download URL for a stored attachment rendition. */
export function attachmentDownloadUrl(
  productId: string,
  attachmentId: string,
): string {
  return `/api/products/${encodeURIComponent(productId)}/ask/attachments/${encodeURIComponent(attachmentId)}`;
}

/**
 * Delete a chat session with its transcript (the trash button / Clear on a
 * historical thread). A running turn is cancelled server-side first.
 */
export async function deleteChatSession(
  productId: string,
  sessionId: string,
): Promise<void> {
  const res = await fetch(
    `/api/products/${encodeURIComponent(productId)}/chat/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE", credentials: "include" },
  );
  if (res.status === 404) throw new ApiError(404, "Chat session not found");
  if (!res.ok) {
    throw new ApiError(res.status, `Failed to delete chat session (${res.status})`);
  }
}
