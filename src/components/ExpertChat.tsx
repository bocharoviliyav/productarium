"use client";

/**
 * Productarium expert agent chat (item 3 — replaces the Long-context RLM panel).
 *
 * Product-scoped chat over ALL artifacts via cognee recall (+ optional fast-rlm
 * deep synthesis) on the backend. Streams answers from POST
 * /api/products/{id}/ask (SSE) and renders the streamed markdown (Mermaid
 * works via the existing <Markdown /> component). "Download as document" hits
 * POST /api/products/{id}/ask/doc and saves the returned .md file.
 *
 * SSE tolerance: the backend may emit either `data: {json}\n\n` frames or raw
 * text chunks. We parse `data:` lines, accept JSON with a {content|delta|text}
 * field, and fall back to appending raw text — so this works whichever stream
 * shape the expert-agent router lands on.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  ChatCircleText,
  DownloadSimple,
  Eraser,
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

interface Turn {
  role: "user" | "assistant";
  content: string;
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
  // RLM routing override: null = Auto (follow admin rlm.expert.mode),
  // false = force standard LLM, true = force RLM (with LLM fallback).
  const [rlmMode, setRlmMode] = useState<boolean | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const lastQuestionRef = useRef<string>("");

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [turns]);

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || streaming) return;
    lastQuestionRef.current = q;
    setTurns((prev) => [...prev, { role: "user", content: q }]);
    setTurns((prev) => [...prev, { role: "assistant", content: "" }]);
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
      let acc = "";

      const append = (text: string) => {
        acc += text;
        setTurns((prev) => {
          const next = prev.slice();
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = { ...last, content: acc };
          }
          return next;
        });
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
            try {
              const obj = JSON.parse(payload);
              const chunk =
                obj?.content ?? obj?.delta ?? obj?.text ?? obj?.message ?? "";
              if (typeof chunk === "string" && chunk) append(chunk);
            } catch {
              // Not JSON — treat the payload as raw text.
              if (payload) append(payload);
            }
          } else if (line.startsWith(":")) {
            // SSE comment / heartbeat — ignore.
          } else {
            // Raw text stream (no SSE framing).
            append(line);
          }
        }
      }
      if (buffer.trim()) {
        const line = buffer.trim();
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          if (payload && payload !== "[DONE]") {
            try {
              const obj = JSON.parse(payload);
              const chunk = obj?.content ?? obj?.delta ?? obj?.text ?? "";
              if (typeof chunk === "string" && chunk) append(chunk);
            } catch {
              append(payload);
            }
          }
        } else {
          append(line);
        }
      }
    } catch (e) {
      if ((e as Error)?.name === "AbortError") {
        // user stopped — keep partial output
      } else {
        const msg = e instanceof Error ? e.message : "Expert chat failed";
        notify({ tone: "error", title: "Expert chat failed", message: msg });
        setTurns((prev) => {
          const next = prev.slice();
          const last = next[next.length - 1];
          if (last && last.role === "assistant" && !last.content) {
            next[next.length - 1] = {
              ...last,
              content: `> ⚠️ ${msg}`,
            };
          }
          return next;
        });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [input, streaming, productId, turns, rlmMode, notify, router]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clear = useCallback(() => {
    setTurns([]);
    setInput("");
  }, []);

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
        className="max-h-[480px] min-h-[160px] flex-1 space-y-4 overflow-y-auto rounded-md border border-divider bg-surface-2 p-4"
      >
        {turns.length === 0 ? null : (
          turns.map((t, i) => (
            <div
              key={i}
              className={cn(
                "flex",
                t.role === "user" ? "justify-end" : "justify-start",
              )}
            >
              <div
                className={cn(
                  "max-w-[88%] rounded-md px-3 py-2 text-sm",
                  t.role === "user"
                    ? "bg-ink text-[var(--button-fg)]"
                    : "bg-surface text-ink border border-divider",
                )}
              >
                {t.role === "assistant" ? (
                  <Markdown content={t.content || "…"} />
                ) : (
                  <p className="whitespace-pre-wrap">{t.content}</p>
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

export default ExpertChat;
