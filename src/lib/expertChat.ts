/**
 * Expert agent SSE stream client (wave B — LangGraph backend).
 *
 * Implements the FIXED chat streaming contract shared with the backend:
 *
 *   data: {"status": "retrieving"|"thinking"|"answering"}
 *   data: {"reasoning": "<model thoughts>"}
 *   data: {"content": "<answer chunk>"}
 *   data: {"tool_call": {"name": "<tool>", "args": {...}}}
 *   data: {"tool_result": {"name": "<tool>", "content": "<result summary>"}}
 *   data: {"error": "<message>"}
 *   data: [DONE]
 *
 * The parser is deliberately tolerant:
 * - unknown JSON frames are ignored (forward compatibility),
 * - a bare `[DONE]` payload terminates the stream,
 * - non-JSON payload lines are surfaced as raw content deltas (legacy
 *   backends that stream plain text keep working),
 * - `{"delta"|"text": ...}` frames are treated as content,
 * - an early `{"session_id": ...}` frame carries the new chat session id.
 */

/**
 * Agent phase reported by the backend via {"status": ...} frames.
 *
 * The deep-research phases (planning / researching / synthesizing, wave E)
 * are additive: they ride the same frame as the base phases and are simply
 * extra members of the union, so older streams keep parsing unchanged.
 */
export type ExpertPhase =
  | "retrieving"
  | "thinking"
  | "answering"
  | "planning"
  | "researching"
  | "synthesizing";

/** View state phase: stream phases plus the terminal "done". */
export type TurnPhase = ExpertPhase | "done";

/** A tool_call frame payload. */
export interface ExpertToolCall {
  name: string;
  args: Record<string, unknown>;
}

/** A tool_result frame payload (short result summary). */
export interface ExpertToolResult {
  name: string;
  content: string;
}

/** Union of all decoded SSE frames from the expert stream. */
export type ExpertSseEvent =
  | { kind: "status"; phase: ExpertPhase }
  | { kind: "reasoning"; text: string }
  | { kind: "content"; text: string }
  | { kind: "tool_call"; call: ExpertToolCall }
  | { kind: "tool_result"; result: ExpertToolResult }
  | { kind: "error"; message: string }
  | { kind: "session"; sessionId: string };

const PHASES: readonly ExpertPhase[] = [
  "retrieving",
  "thinking",
  "answering",
  "planning",
  "researching",
  "synthesizing",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Decode one `data:` payload string into a typed event, or null to skip it. */
export function parseExpertSsePayload(payload: string): ExpertSseEvent | null {
  const text = payload.trim();
  if (!text || text === "[DONE]") return null;

  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch {
    // Plain text frame — treat it as an answer content delta.
    return { kind: "content", text };
  }

  if (!isRecord(obj)) return null;

  if (typeof obj.reasoning === "string" && obj.reasoning) {
    return { kind: "reasoning", text: obj.reasoning };
  }
  if (typeof obj.status === "string" && PHASES.includes(obj.status as ExpertPhase)) {
    return { kind: "status", phase: obj.status as ExpertPhase };
  }
  if (typeof obj.error === "string" && obj.error) {
    return { kind: "error", message: obj.error };
  }
  if (typeof obj.session_id === "string" && obj.session_id) {
    return { kind: "session", sessionId: obj.session_id };
  }
  const toolCall = obj.tool_call;
  if (isRecord(toolCall) && typeof toolCall.name === "string" && toolCall.name) {
    return {
      kind: "tool_call",
      call: {
        name: toolCall.name,
        args: isRecord(toolCall.args) ? toolCall.args : {},
      },
    };
  }
  const toolResult = obj.tool_result;
  if (isRecord(toolResult) && typeof toolResult.name === "string" && toolResult.name) {
    return {
      kind: "tool_result",
      result: {
        name: toolResult.name,
        // The contract promises a short string summary; tolerate missing
        // values and non-string payloads rather than dropping the frame.
        content: typeof toolResult.content === "string" ? toolResult.content : "",
      },
    };
  }
  // Content / legacy delta / text frames.
  const chunk = obj.content ?? obj.delta ?? obj.text;
  if (typeof chunk === "string" && chunk) {
    return { kind: "content", text: chunk };
  }
  return null;
}

/** Async iterator of decoded expert events from a fetch Response body. */
export async function* readExpertStream(
  response: Response,
): AsyncGenerator<ExpertSseEvent> {
  if (!response.body) {
    throw new Error("Expert stream has no body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Process complete lines; keep the trailing partial line buffered.
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const raw of lines) {
        const event = decodeSseLine(raw);
        if (event) yield event;
      }
    }
    // Flush any trailing line that arrived without a newline.
    const trailing = decodeSseLine(buffer);
    if (trailing) yield trailing;
  } finally {
    reader.releaseLock();
  }
}

/**
 * Decode one raw stream line.
 *
 * `data:`-prefixed lines carry the typed contract payload. `event:` lines and
 * SSE comments (`:` heartbeat) are ignored. Bare (non-SSE) lines are treated
 * as raw content so plain-text backends still render.
 */
function decodeSseLine(raw: string): ExpertSseEvent | null {
  const line = raw.trimEnd();
  if (!line) return null;
  if (line.startsWith(":")) return null; // SSE comment / heartbeat.
  if (line.startsWith("event:")) return null; // event name hints are unused.
  if (line.startsWith("data:")) {
    return parseExpertSsePayload(line.slice(5));
  }
  // Raw (unframed) text — legacy plain-text streaming.
  return { kind: "content", text: line };
}

/* ------------------------------------------------------------------ */
/* Tool event view models                                              */
/* ------------------------------------------------------------------ */

/**
 * One compact tool activity entry rendered as a chip row between the
 * reasoning panel and the answer content. Call chips are upgraded in place
 * when the matching result arrives (matched by tool name, FIFO).
 */
export interface ToolEvent {
  id: number;
  name: string;
  args: Record<string, unknown>;
  /** Short result summary once the tool finished; null while running. */
  result: string | null;
}

let toolEventSeq = 0;

/** Build a fresh running tool event (call phase). */
export function makeToolEvent(call: ExpertToolCall): ToolEvent {
  toolEventSeq += 1;
  return { id: toolEventSeq, name: call.name, args: call.args, result: null };
}

/** Human-friendly single-line rendering of tool arguments for a chip. */
export function formatToolArgs(args: Record<string, unknown>, maxLen = 48): string {
  const text = Object.entries(args)
    .map(([k, v]) => `${k}=${formatToolArgValue(v)}`)
    .join(" ");
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 1)}…`;
}

function formatToolArgValue(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}
