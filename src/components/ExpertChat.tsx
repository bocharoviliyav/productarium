"use client";

/**
 * Productarium expert agent chat — streaming + reasoning + phase-aware loader.
 *
 * Product-scoped chat over ALL artifacts via cognee recall (+ optional fast-rlm
 * deep synthesis) on the backend. Streams answers from POST
 * /api/products/{id}/ask (SSE) and renders the streamed markdown (Mermaid
 * works via the existing <Markdown /> component). "Download as document" hits
 * POST /api/products/{id}/ask/doc and saves the returned .md file.
 *
 * SSE event types (typed frames from the backend router):
 * - {"content": "..."}     — answer text delta
 * - {"reasoning": "..."}   — reasoning/thinking trace delta (Qwen3/DeepSeek)
 * - {"status": "retrieving"|"thinking"|"answering"} — phase indicator
 * - {"error": "..."}        — error message
 * - [DONE]                  — stream end
 *
 * Backward-compatible: if the backend emits plain text or untyped {content}
 * frames, they are appended to the answer content.
 *
 * UX features (hand-built, no external chat libraries):
 * - Phase-aware loader: "Retrieving knowledge…" / "Thinking…" / "Generating…"
 *   with a 3-dot pulse animation.
 * - Collapsible reasoning panel: auto-expanded while reasoning streams,
 *   auto-collapses when content starts. User can toggle manually.
 * - Streaming cursor: pulsing ▍ appended to content while streaming.
 * - Smart auto-scroll: smooth scroll to bottom on new content, but pauses
 *   when the user scrolls up.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Brain,
  CaretDown,
  ChatCircleText,
  DownloadSimple,
  Eraser,
  MagnifyingGlass,
  Spinner,
  StopCircle,
} from "@phosphor-icons/react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { Button, cn, Textarea } from "@/components/ui";

const Markdown = dynamic(() => import("@/components/Markdown"), {
  ssr: false,
  loading: () => <div className="text-xs text-muted">Rendering…</div>,
});

interface ExpertChatProps {
  productId: string;
  className?: string;
}

type Phase = "retrieving" | "thinking" | "answering" | "done";

interface Turn {
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  phase: Phase;
  streaming: boolean;
  reasoningOpen: boolean;
  /** Whether the user manually toggled the reasoning panel. */
  reasoningTouched: boolean;
}

const EMPTY_TURN: Omit<Turn, "role"> = {
  content: "",
  reasoning: "",
  phase: "retrieving",
  streaming: true,
  reasoningOpen: true,
  reasoningTouched: false,
};

export function ExpertChat({ productId, className }: ExpertChatProps) {
  const { messages } = useLanguage();
  const { notify } = useNotifications();
  const router = useRouter();
  const t = messages?.expert ?? {};
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [downloading, setDownloading] = useState(false);
  // RLM routing override: null = Auto (follow admin rlm.expert.mode),
  // false = force standard LLM, true = force RLM (with LLM fallback).
  const [rlmMode, setRlmMode] = useState<boolean | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const lastQuestionRef = useRef<string>("");
  // Track whether the user has scrolled up — pause auto-scroll if so.
  const userScrolledUpRef = useRef(false);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

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
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 60;
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
    (phase: Phase) => {
      updateLastTurn((last) => ({ ...last, phase }));
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
      const res = await fetch(`/api/products/${productId}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        credentials: "include",
        body: JSON.stringify({
          query: q,
          messages: turns.map((t) => ({ role: t.role, content: t.content })),
          stream: true,
          use_rlm: rlmMode,
        }),
        signal: controller.signal,
      });
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}`);
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Expert request failed (${res.status})`);
      }
      if (!res.body) throw new Error("No response stream");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const processPayload = (payload: string) => {
        if (payload === "[DONE]") return;
        try {
          const obj = JSON.parse(payload);
          // Typed SSE frames: {content} / {reasoning} / {status} / {error}
          if (typeof obj.reasoning === "string" && obj.reasoning) {
            appendReasoning(obj.reasoning);
            return;
          }
          if (typeof obj.status === "string" && obj.status) {
            setPhase(obj.status as Phase);
            return;
          }
          if (typeof obj.error === "string" && obj.error) {
            updateLastTurn((last) => ({
              ...last,
              content: last.content || `> ⚠️ ${obj.error}`,
              streaming: false,
              phase: "done",
            }));
            return;
          }
          const chunk = obj?.content ?? obj?.delta ?? obj?.text ?? "";
          if (typeof chunk === "string" && chunk) {
            appendContent(chunk);
            setPhase("answering");
          }
        } catch {
          // Not JSON — treat the payload as raw text content.
          if (payload) {
            appendContent(payload);
            setPhase("answering");
          }
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Process complete lines; keep the trailing partial line in buffer.
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const raw of lines) {
          const line = raw.trimEnd();
          if (!line) continue;
          if (line.startsWith("data:")) {
            const payload = line.slice(5).trim();
            if (payload === "[DONE]") {
              buffer = "";
              continue;
            }
            processPayload(payload);
          } else if (line.startsWith(":")) {
            // SSE comment / heartbeat — ignore.
          } else {
            // Raw text stream (no SSE framing).
            appendContent(line);
            setPhase("answering");
          }
        }
      }
      // Process any remaining buffered text.
      if (buffer.trim()) {
        const line = buffer.trim();
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          if (payload && payload !== "[DONE]") {
            processPayload(payload);
          }
        } else {
          appendContent(line);
          setPhase("answering");
        }
      }
    } catch (e) {
      if ((e as Error)?.name === "AbortError") {
        // user stopped — keep partial output
      } else {
        const msg = e instanceof Error ? e.message : "Expert chat failed";
        notify({ tone: "error", title: "Expert chat failed", message: msg });
        updateLastTurn((last) => {
          if (last.content) return last; // keep partial output if any
          return { ...last, content: `> ⚠️ ${msg}`, streaming: false, phase: "done" };
        });
      }
    } finally {
      finishTurn();
      setStreaming(false);
      abortRef.current = null;
    }
  }, [
    input,
    streaming,
    productId,
    turns,
    rlmMode,
    notify,
    router,
    appendContent,
    appendReasoning,
    setPhase,
    finishTurn,
    updateLastTurn,
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
        body: JSON.stringify({ query: q, use_rlm: rlmMode }),
      });
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}`);
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Document request failed (${res.status})`);
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
  }, [downloading, input, productId, rlmMode, notify, router]);

  return (
    <div className={cn("flex flex-col", className)}>
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
          </div>
          <div className="flex items-center gap-2">
            <div
              role="group"
              aria-label={t.engine ?? "Answer engine"}
              className="flex items-center rounded-md border border-divider bg-surface-2 p-0.5"
            >
              {([
                { value: null, label: t.engineAuto ?? "Auto" },
                { value: false, label: t.engineLlm ?? "LLM" },
                { value: true, label: t.engineRlm ?? "RLM" },
              ] as const).map((opt) => (
                <button
                  key={String(opt.value)}
                  type="button"
                  onClick={() => setRlmMode(opt.value)}
                  disabled={streaming || downloading}
                  title={
                    opt.value === null
                      ? (t.engineAutoHint ??
                        "Auto: follow admin RLM/LLM mode (RLM for large context, else LLM)")
                      : opt.value === false
                        ? (t.engineLlmHint ?? "LLM: standard model, faster")
                        : (t.engineRlmHint ??
                          "RLM: recursive reasoning for large context, falls back to LLM on failure")
                  }
                  className={cn(
                    "rounded px-2 py-1 text-xs font-medium transition-colors",
                    "disabled:cursor-not-allowed disabled:opacity-50",
                    rlmMode === opt.value
                      ? "bg-ink text-[var(--button-fg)]"
                      : "text-muted hover:text-ink",
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
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

  return (
    <div className="space-y-2">
      {/* Phase-aware loader (shown before content arrives) */}
      {showLoader && (
        <PhaseLoader phase={phase} t={t} />
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
              {streaming && phase === "thinking" && (
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
              {streaming && phase === "thinking" && (
                <span className="stream-cursor">▍</span>
              )}
            </div>
          )}
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
    </div>
  );
}

// --- 3-dot pulse loader ------------------------------------------------------

function PhaseLoader({ phase, t }: { phase: Phase; t: Record<string, string> }) {
  let label = t.generating ?? "Generating…";
  let icon = <Spinner size={14} />;

  if (phase === "retrieving") {
    label = t.retrievingKnowledge ?? "Retrieving knowledge…";
    icon = <MagnifyingGlass size={14} weight="regular" />;
  } else if (phase === "thinking") {
    label = t.thinking ?? "Thinking…";
    icon = <Brain size={14} weight="regular" />;
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
