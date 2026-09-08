"use client";

/**
 * Productarium expert agent chat — wave B agent backend.
 *
 * Product-scoped chat over the LangGraph expert agent. Streams the FIXED SSE
 * contract (see src/lib/expertChat.ts) from POST /api/products/{id}/ask:
 * status → phase loader, reasoning → collapsible "Reasoning" panel (mono,
 * muted), content → markdown answer, tool_call/tool_result → compact chips,
 * error → inline error row. "Download as document" hits POST
 * /api/products/{id}/ask/doc and saves the returned .md file.
 *
 * Chat history: sessions are persisted server-side. On mount the component
 * lists GET /api/products/{id}/chat/sessions; picking a session loads
 * GET .../chat/sessions/{sessionId}/messages. Sends carry the active
 * session_id (a brand-new chat sends none). The id of a newly created
 * session is read BOTH from the X-Session-Id response header AND from an
 * early SSE {"session_id": ...} frame — whichever arrives first wins.
 *
 * UX features (hand-built, no external chat libraries):
 * - Phase-aware loader: "Retrieving knowledge…" / "Thinking…" / "Generating…"
 *   with a 3-dot pulse animation.
 * - Collapsible reasoning panel: auto-expanded while reasoning streams,
 *   auto-collapses when content starts. User can toggle manually.
 * - Compact tool chips that flip from spinner to a short result summary.
 * - Inline error row (red tag palette, no toast spam for stream errors).
 * - Streaming cursor: pulsing ▍ appended to content while streaming.
 * - Smart auto-scroll: smooth scroll to bottom on new content, but pauses
 *   when the user scrolls up.
 *
 * The sessions feature degrades gracefully: if the history endpoints are not
 * deployed yet (404) the strip hides and the chat behaves like before.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Binoculars,
  Brain,
  CaretDown,
  ChatCircleText,
  DownloadSimple,
  Eraser,
  MagnifyingGlass,
  Sparkle,
  Spinner,
  StopCircle,
  WarningCircle,
} from "@phosphor-icons/react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { Button, cn, Textarea } from "@/components/ui";
import { SessionStrip } from "@/components/expert/SessionStrip";
import { ToolEventChip } from "@/components/expert/ToolEventChip";
import {
  makeToolEvent,
  readExpertStream,
  type ExpertPhase,
  type ToolEvent,
  type TurnPhase,
} from "@/lib/expertChat";
import {
  ApiError,
  fetchChatMessages,
  fetchChatSessions,
  type ChatMessage as ChatHistoryMessage,
  type ChatSession,
} from "@/lib/chatSessions";

const Markdown = dynamic(() => import("@/components/Markdown"), {
  ssr: false,
  loading: () => <div className="text-xs text-muted">Rendering…</div>,
});

interface ExpertChatProps {
  productId: string;
  className?: string;
}

interface Turn {
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  phase: TurnPhase;
  streaming: boolean;
  reasoningOpen: boolean;
  /** Whether the user manually toggled the reasoning panel. */
  reasoningTouched: boolean;
  /** Tool activity chips (assistant turns only). */
  tools: ToolEvent[];
  /** Inline stream error (assistant turns only). */
  error: string | null;
}

const EMPTY_TURN: Omit<Turn, "role"> = {
  content: "",
  reasoning: "",
  phase: "retrieving",
  streaming: true,
  reasoningOpen: true,
  reasoningTouched: false,
  tools: [],
  error: null,
};

/**
 * Map persisted session history onto chat turns.
 *
 * User/assistant rows become bubbles; ``role='tool'`` rows (short tool
 * summary + name + args) are attached as completed chips to the NEXT
 * assistant bubble so the transcript reads like a live stream.
 */
function historyToTurns(history: ChatHistoryMessage[]): Turn[] {
  const turns: Turn[] = [];
  let pendingTools: ToolEvent[] = [];
  for (const message of history) {
    if (message.role === "tool") {
      pendingTools = [
        ...pendingTools,
        {
          ...makeToolEvent({
            name: message.toolName ?? "tool",
            args: message.toolArgs ?? {},
          }),
          result: message.content,
        },
      ];
      continue;
    }
    turns.push({
      role: message.role === "assistant" ? "assistant" : "user",
      ...EMPTY_TURN,
      content: message.content,
      streaming: false,
      reasoningOpen: false,
      phase: "done",
      tools: message.role === "assistant" ? pendingTools : [],
    });
    if (message.role === "assistant") pendingTools = [];
  }
  return turns;
}

export function ExpertChat({ productId, className }: ExpertChatProps) {
  const { messages } = useLanguage();
  const { notify } = useNotifications();
  const router = useRouter();
  const t = messages?.expert ?? {};
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [downloading, setDownloading] = useState(false);
  // Deep Research mode (wave E): sends {deep_research: true} so the backend
  // routes the query through the planner → researcher → synthesizer graph.
  const [deepResearch, setDeepResearch] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const lastQuestionRef = useRef<string>("");
  // Track whether the user has scrolled up — pause auto-scroll if so.
  const userScrolledUpRef = useRef(false);

  // --- Chat sessions (persistent history) -------------------------------
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const activeSessionIdRef = useRef<string | null>(null);
  const sessionsAvailableRef = useRef(false);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  const loadSessions = useCallback(async () => {
    try {
      const list = await fetchChatSessions(productId);
      sessionsAvailableRef.current = list.length > 0;
      setSessions(list);
      // Pick the most recently updated session as the default.
      const latest = [...list].sort((a, b) => {
        const ta = Date.parse(a.updatedAt ?? a.createdAt ?? "") || 0;
        const tb = Date.parse(b.updatedAt ?? b.createdAt ?? "") || 0;
        return tb - ta;
      })[0];
      if (latest && activeSessionIdRef.current === null) {
        activeSessionIdRef.current = latest.id;
        setActiveSessionId(latest.id);
      }
    } catch {
      // 401 redirects at the ask boundary; otherwise hide the strip.
      sessionsAvailableRef.current = false;
    } finally {
      setSessionsLoading(false);
    }
  }, [productId]);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  /** Select an existing session and load its persisted history. */
  const selectSession = useCallback(
    async (sessionId: string) => {
      if (streaming) abortRef.current?.abort();
      setActiveSessionId(sessionId);
      activeSessionIdRef.current = sessionId;
      setTurns([]);
      setSessionsLoading(true);
      try {
        const history = await fetchChatMessages(productId, sessionId);
        setTurns(historyToTurns(history));
        // Auto-scroll to the newest history message.
        userScrolledUpRef.current = false;
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) {
          router.replace(`/login?next=/products/${productId}`);
          return;
        }
        notify({
          tone: "error",
          title: t.historyFailedTitle ?? "History",
          message: e instanceof Error ? e.message : "Failed to load history",
        });
      } finally {
        setSessionsLoading(false);
      }
    },
    [streaming, productId, router, notify, t.historyFailedTitle],
  );

  /** Start a brand-new chat (unsent — the session is created on first ask). */
  const newChat = useCallback(() => {
    if (streaming) abortRef.current?.abort();
    setActiveSessionId(null);
    activeSessionIdRef.current = null;
    setTurns([]);
    setInput("");
  }, [streaming]);

  // --- Auto-scroll: only scroll down if the user hasn't scrolled up.
  const scrollToBottom = useCallback(() => {
    if (userScrolledUpRef.current || !scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [turns, scrollToBottom]);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // If the user is near the bottom (< 60px from the bottom), resume auto-scroll.
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    userScrolledUpRef.current = !nearBottom;
  }, []);

  // --- Update the last assistant turn.
  const updateLastTurn = useCallback(
    (updater: (prev: Turn) => Turn) => {
      setTurns((prev) => {
        const next = prev.slice();
        const last = next[next.length - 1];
        if (last && last.role === "assistant") {
          next[next.length - 1] = updater(last);
        }
        return next;
      });
    },
    [],
  );

  const appendContent = useCallback(
    (text: string) => {
      updateLastTurn((last) => {
        const wasStreaming = last.streaming;
        return {
          ...last,
          content: last.content + text,
          streaming: true,
          // Auto-collapse reasoning when content starts flowing (unless user
          // manually toggled it open).
          reasoningOpen: wasStreaming
            ? last.reasoningTouched
              ? last.reasoningOpen
              : false
            : last.reasoningOpen,
        };
      });
    },
    [updateLastTurn],
  );

  const appendReasoning = useCallback(
    (text: string) => {
      updateLastTurn((last) => ({
        ...last,
        reasoning: last.reasoning + text,
        // Auto-expand reasoning panel while streaming (unless user manually
        // collapsed it).
        reasoningOpen: last.reasoningTouched ? last.reasoningOpen : true,
      }));
    },
    [updateLastTurn],
  );

  const setPhase = useCallback(
    (phase: ExpertPhase) => {
      updateLastTurn((last) => ({ ...last, phase }));
    },
    [updateLastTurn],
  );

  /** Attach a running tool chip; the result upgrades it in place (FIFO). */
  const addToolCall = useCallback(
    (name: string, args: Record<string, unknown>) => {
      updateLastTurn((last) => ({
        ...last,
        tools: [...last.tools, makeToolEvent({ name, args })],
      }));
    },
    [updateLastTurn],
  );

  const addToolResult = useCallback(
    (name: string, content: string) => {
      updateLastTurn((last) => {
        // Upgrade the oldest matching running tool, if any; else append a
        // standalone completed chip so no result is silently dropped.
        const idx = last.tools.findIndex((e) => e.name === name && e.result === null);
        if (idx === -1) {
          return {
            ...last,
            tools: [
              ...last.tools,
              { ...makeToolEvent({ name, args: {} }), result: content },
            ],
          };
        }
        const tools = last.tools.slice();
        tools[idx] = { ...tools[idx], result: content };
        return { ...last, tools };
      });
    },
    [updateLastTurn],
  );

  const setTurnError = useCallback(
    (message: string) => {
      updateLastTurn((last) => ({
        ...last,
        error: last.error ?? message,
        streaming: false,
        phase: "done",
      }));
    },
    [updateLastTurn],
  );

  const finishTurn = useCallback(() => {
    updateLastTurn((last) => ({
      ...last,
      streaming: false,
      phase: "done",
    }));
  }, [updateLastTurn]);

  /**
   * Register a session id captured mid-stream (SSE frame or header) when the
   * current ask started a brand-new chat.
   */
  const adoptSessionId = useCallback((sessionId: string) => {
    if (!sessionId) return;
    if (activeSessionIdRef.current !== null) return; // existing session
    activeSessionIdRef.current = sessionId;
    setActiveSessionId(sessionId);
    // Optimistically add the pill so the user can see/keep the new session.
    setSessions((prev) =>
      prev.some((s) => s.id === sessionId)
        ? prev
        : [...prev, { id: sessionId, title: null, createdAt: null, updatedAt: null }],
    );
    sessionsAvailableRef.current = true;
  }, []);

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || streaming) return;
    lastQuestionRef.current = q;
    // Reset scroll state for a new question.
    userScrolledUpRef.current = false;
    setTurns([
      ...turns,
      { role: "user", ...EMPTY_TURN, content: q, streaming: false },
      { role: "assistant", ...EMPTY_TURN },
    ]);
    setStreaming(true);
    setInput("");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const body: Record<string, unknown> = {
        query: q,
        stream: true,
        // Dialog history for backends that do not resolve it from the
        // session (harmless extra field otherwise — ignored server-side).
        messages: turns
          .filter((turn) => turn.content)
          .map((turn) => ({ role: turn.role, content: turn.content })),
      };
      // Carry the active session so the agent keeps its persistent context;
      // a brand-new chat sends none (the backend creates the session).
      const sessionId = activeSessionIdRef.current;
      if (sessionId) body.session_id = sessionId;
      // Deep Research flag — additive; ignored by backends without the graph.
      if (deepResearch) body.deep_research = true;

      const res = await fetch(`/api/products/${productId}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        credentials: "include",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}`);
        return;
      }
      if (res.status === 429) {
        // Per-user rate limit (P1-17): human message + Retry-After.
        const detail = await res.json().catch(() => ({}));
        const ra = res.headers.get("Retry-After");
        const wait = ra && /^\d+$/.test(ra) ? Number(ra) : null;
        throw new Error(
          `${
            (detail as { detail?: string })?.detail ||
            "Too many requests — please wait a moment and try again."
          }${wait ? ` (retry in ~${wait}s)` : ""}`,
        );
      }
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(
          (detail as { detail?: string })?.detail ||
            `Expert request failed (${res.status})`,
        );
      }

      // Session id variant 1: response header.
      const headerSession = res.headers.get("X-Session-Id");
      if (headerSession) adoptSessionId(headerSession);

      for await (const event of readExpertStream(res)) {
        switch (event.kind) {
          case "session":
            // Session id variant 2: early SSE frame.
            adoptSessionId(event.sessionId);
            break;
          case "status":
            setPhase(event.phase);
            break;
          case "reasoning":
            appendReasoning(event.text);
            break;
          case "content":
            appendContent(event.text);
            setPhase("answering");
            break;
          case "tool_call":
            addToolCall(event.call.name, event.call.args);
            break;
          case "tool_result":
            addToolResult(event.result.name, event.result.content);
            break;
          case "error":
            setTurnError(event.message);
            break;
        }
      }
    } catch (e) {
      if ((e as Error)?.name === "AbortError") {
        // user stopped — keep partial output
      } else {
        const msg = e instanceof Error ? e.message : "Expert chat failed";
        notify({ tone: "error", title: "Expert chat failed", message: msg });
        setTurnError(msg);
      }
    } finally {
      finishTurn();
      setStreaming(false);
      abortRef.current = null;
      // Refresh the session strip (titles/timestamps) after the turn.
      if (activeSessionIdRef.current !== null) void loadSessions();
    }
  }, [
    input,
    streaming,
    productId,
    turns,
    deepResearch,
    notify,
    router,
    appendContent,
    appendReasoning,
    setPhase,
    addToolCall,
    addToolResult,
    setTurnError,
    finishTurn,
    adoptSessionId,
    loadSessions,
  ]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clear = useCallback(() => {
    setTurns([]);
    setInput("");
  }, []);

  const toggleReasoning = useCallback(
    (idx: number) => {
      setTurns((prev) => {
        const next = prev.slice();
        const turn = next[idx];
        if (turn && turn.role === "assistant") {
          next[idx] = {
            ...turn,
            reasoningOpen: !turn.reasoningOpen,
            reasoningTouched: true,
          };
        }
        return next;
      });
    },
    [],
  );

  const downloadDoc = useCallback(async () => {
    const q = lastQuestionRef.current.trim() || input.trim();
    if (!q || downloading) return;
    setDownloading(true);
    try {
      const res = await fetch(`/api/products/${productId}/ask/doc`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ query: q }),
      });
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}`);
        return;
      }
      if (res.status === 429) {
        // Per-user rate limit (P1-17): human message + Retry-After.
        const detail = await res.json().catch(() => ({}));
        const ra = res.headers.get("Retry-After");
        const wait = ra && /^\d+$/.test(ra) ? Number(ra) : null;
        throw new Error(
          `${
            (detail as { detail?: string })?.detail ||
            "Too many requests — please wait a moment and try again."
          }${wait ? ` (retry in ~${wait}s)` : ""}`,
        );
      }
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(
          (detail as { detail?: string })?.detail ||
            `Document request failed (${res.status})`,
        );
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      a.download = `productarium-expert-${stamp}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      notify({
        tone: "error",
        title: "Download failed",
        message: e instanceof Error ? e.message : "Download failed",
      });
    } finally {
      setDownloading(false);
    }
  }, [downloading, input, productId, notify, router]);

  const showSessions =
    sessionsAvailableRef.current || sessions.length > 0 || !sessionsLoading;

  return (
    <div className={cn("flex flex-col", className)}>
      {/* Session strip (hidden while the backend has no history endpoints) */}
      {showSessions && (
        <SessionStrip
          sessions={sessions}
          activeSessionId={activeSessionId}
          loading={sessionsLoading}
          text={{
            newChat: t.newChat ?? "New chat",
            sessionFallback: t.sessionFallback ?? "Session",
            loading: t.sessionsLoading ?? "",
          }}
          onSelect={(id) => void selectSession(id)}
          onNewChat={newChat}
        />
      )}

      {/* Conversation */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="max-h-[480px] min-h-[160px] flex-1 space-y-4 overflow-y-auto rounded-md border border-divider bg-surface-2 p-4"
      >
        {turns.length === 0 ? (
          <div className="flex h-full min-h-[120px] items-center justify-center text-center">
            <p className="max-w-sm text-sm text-muted">
              {t.emptyHint ??
                "Ask the expert anything about this product. Answers are grounded in its indexed artifacts via the knowledge graph."}
            </p>
          </div>
        ) : (
          turns.map((turn, i) => (
            <div
              key={i}
              className={cn(
                "flex",
                turn.role === "user" ? "justify-end" : "justify-start",
              )}
            >
              <div
                className={cn(
                  "max-w-[88%] rounded-md px-3 py-2 text-sm",
                  turn.role === "user"
                    ? "bg-ink text-[var(--button-fg)]"
                    : "bg-surface text-ink border border-divider",
                )}
              >
                {turn.role === "assistant" ? (
                  <AssistantContent
                    turn={turn}
                    t={t}
                    onToggleReasoning={() => toggleReasoning(i)}
                  />
                ) : (
                  <p className="whitespace-pre-wrap">{turn.content}</p>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Input + actions */}
      <div className="mt-3">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            t.placeholder ??
            "Ask the expert: map the data flow across services, summarize the API contract, draft an on-call runbook…"
          }
          rows={3}
          className="font-sans"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <div className="mt-2 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Button
              type="button"
              onClick={streaming ? stop : send}
              disabled={!streaming && !input.trim()}
            >
              {streaming ? (
                <>
                  <StopCircle size={16} weight="fill" />
                  {t.stop ?? "Stop"}
                </>
              ) : (
                <>
                  <ArrowUp size={16} weight="bold" />
                  {t.askExpert ?? "Ask expert"}
                </>
              )}
            </Button>
            <Button
              type="button"
              variant="ghost"
              onClick={downloadDoc}
              disabled={downloading || (!lastQuestionRef.current && !input.trim())}
              title={t.downloadAsDocument ?? "Download the answer as a markdown document"}
            >
              {downloading ? <Spinner /> : <DownloadSimple size={16} weight="regular" />}
              {t.downloadAsDocument ?? "Download as document"}
            </Button>
            {/* Deep Research toggle — same quiet button language, highlighted
                (blue tag palette + filled icon) while active. */}
            <Button
              type="button"
              variant={deepResearch ? "subtle" : "ghost"}
              onClick={() => setDeepResearch((v) => !v)}
              aria-pressed={deepResearch}
              title={t.deepResearchHint ?? ""}
              className={deepResearch ? "text-tag-blue-fg" : undefined}
            >
              <Binoculars size={16} weight={deepResearch ? "fill" : "regular"} />
              {t.deepResearch ?? "Deep Research"}
            </Button>
          </div>
          <div className="flex items-center gap-2">
            {turns.length > 0 && (
              <Button type="button" variant="ghost" onClick={clear}>
                <Eraser size={14} weight="regular" />
                {t.clear ?? "Clear"}
              </Button>
            )}
            <span className="hidden items-center gap-1 text-xs text-muted sm:flex">
              <ChatCircleText size={13} weight="regular" />
              ⌘⏎
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Phase-aware loader + reasoning panel + content rendering ---------------

interface AssistantContentProps {
  turn: Turn;
  t: Record<string, string>;
  onToggleReasoning: () => void;
}

function AssistantContent({ turn, t, onToggleReasoning }: AssistantContentProps) {
  const { phase, streaming, reasoning, content, reasoningOpen } = turn;
  const showLoader = streaming && !content;
  const showReasoning = reasoning.length > 0;
  // Tool currently running (no result yet) — surfaced in the deep-research
  // loader label ("Researching: <tool>…").
  const runningTool = turn.tools.find((e) => e.result === null)?.name;
  // Any streaming phase that precedes the final answer — used for the
  // reasoning live cursor (covers the deep-research phases too).
  const preAnswer =
    phase === "thinking" ||
    phase === "planning" ||
    phase === "researching" ||
    phase === "synthesizing";

  return (
    <div className="space-y-2">
      {/* Phase-aware loader (shown before content arrives) */}
      {showLoader && (
        <PhaseLoader phase={phase} t={t} activeTool={runningTool} />
      )}

      {/* Reasoning panel */}
      {showReasoning && (
        <div className="reasoning-panel rounded border border-divider bg-surface-2 p-2">
          <button
            type="button"
            onClick={onToggleReasoning}
            className="flex w-full items-center gap-1.5 text-xs font-medium text-muted transition-colors hover:text-ink"
          >
            <Brain size={13} weight="regular" />
            <span>
              {t.reasoning ?? "Reasoning"}
              {streaming && preAnswer && (
                <span className="ml-1 reasoning-stream-dot" />
              )}
            </span>
            <CaretDown
              size={12}
              weight="bold"
              className={cn(
                "ml-auto transition-transform",
                reasoningOpen ? "rotate-0" : "-rotate-90",
              )}
            />
          </button>
          {reasoningOpen && (
            <div className="mt-1.5 max-h-[200px] overflow-y-auto border-l-2 border-divider pl-2.5 font-mono text-[12px] leading-relaxed text-muted">
              {reasoning}
              {streaming && preAnswer && (
                <span className="stream-cursor">▍</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Tool activity chips */}
      {turn.tools.length > 0 && (
        <div className="flex flex-col items-start gap-1">
          {turn.tools.map((event) => (
            <ToolEventChip key={event.id} event={event} />
          ))}
        </div>
      )}

      {/* Answer content */}
      {content ? (
        <div className="streaming-content">
          <Markdown content={content} />
          {streaming && phase === "answering" && (
            <span className="stream-cursor">▍</span>
          )}
        </div>
      ) : !showLoader ? (
        <span className="text-muted">…</span>
      ) : null}

      {/* Inline stream error */}
      {turn.error && (
        <div
          role="alert"
          className="flex items-start gap-1.5 rounded-md border border-tag-red-bg bg-tag-red-bg px-2 py-1.5 text-xs text-tag-red-fg"
        >
          <WarningCircle size={13} weight="fill" className="mt-0.5 shrink-0" aria-hidden />
          <span className="min-w-0 break-words">{turn.error}</span>
        </div>
      )}
    </div>
  );
}

// --- 3-dot pulse loader ------------------------------------------------------

function PhaseLoader({
  phase,
  t,
  activeTool,
}: {
  phase: TurnPhase;
  t: Record<string, string>;
  activeTool?: string;
}) {
  let label = t.generating ?? "Generating…";
  let icon = <Spinner size={14} />;

  if (phase === "retrieving") {
    label = t.retrievingKnowledge ?? "Retrieving knowledge…";
    icon = <MagnifyingGlass size={14} weight="regular" />;
  } else if (phase === "thinking") {
    label = t.thinking ?? "Thinking…";
    icon = (
      <span className="inline-flex items-center gap-1">
        <Brain size={14} weight="regular" />
        <Sparkle size={10} weight="fill" />
      </span>
    );
  } else if (phase === "planning") {
    // Deep research: planner iteration.
    label = t.planning ?? "Planning…";
    icon = <Brain size={14} weight="regular" />;
  } else if (phase === "researching") {
    // Deep research: researcher iteration — name the running tool when known.
    label = activeTool
      ? (t.researchingTool ?? "Researching: {tool}…").replace("{tool}", activeTool)
      : (t.researching ?? "Researching…");
    icon = <MagnifyingGlass size={14} weight="regular" />;
  } else if (phase === "synthesizing") {
    // Deep research: final synthesis pass.
    label = t.synthesizing ?? "Synthesizing…";
    icon = <Sparkle size={14} weight="fill" />;
  }

  return (
    <div className="flex items-center gap-2 text-xs text-muted">
      <span className="inline-flex items-center gap-1.5">
        {icon}
        <span>{label}</span>
      </span>
      <span className="thinking-dots" aria-hidden>
        <span />
        <span />
        <span />
      </span>
    </div>
  );
}

export default ExpertChat;
