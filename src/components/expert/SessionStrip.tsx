"use client";

/**
 * Chat session strip for the expert agent chat (wave B).
 *
 * Horizontally-scrollable pill row above the conversation: "New chat" plus
 * one pill per persisted session. The active session is filled (ink), others
 * are quiet surface pills. Session pills show the title (or a localized
 * "Session" fallback plus a short id) and a relative time hint.
 *
 * minimalist-ui: 1px dividers, surface fills, mono for ids, no shadows.
 */

import { ChatCircle, Plus } from "@phosphor-icons/react";
import type { ChatSession } from "@/lib/chatSessions";
import { cn } from "@/components/ui";

export interface SessionStripText {
  newChat: string;
  sessionFallback: string;
  loading?: string;
}

/** Format a short relative-time label for a session pill (client only). */
function relativeTime(iso: string | null): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const diff = Date.now() - t;
  const minutes = Math.round(diff / 60000);
  if (minutes < 1) return "";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.round(hours / 24)}d`;
}

export function SessionStrip({
  sessions,
  activeSessionId,
  loading,
  text,
  onSelect,
  onNewChat,
}: {
  sessions: ChatSession[];
  /** Active session id, or null when composing a brand-new chat. */
  activeSessionId: string | null;
  loading: boolean;
  text: SessionStripText;
  onSelect: (sessionId: string) => void;
  onNewChat: () => void;
}) {
  const isEmpty = sessions.length === 0;
  return (
    <div className="mb-3 flex items-center gap-1.5 overflow-x-auto pb-1">
      <button
        type="button"
        onClick={onNewChat}
        className={cn(
          "inline-flex shrink-0 items-center gap-1 rounded-full border border-divider px-2.5 py-1",
          "text-xs font-medium transition-colors",
          activeSessionId === null
            ? "bg-[var(--button-bg)] text-[var(--button-fg)]"
            : "bg-surface text-ink hover:bg-surface-2",
        )}
      >
        <Plus size={12} weight="bold" />
        {text.newChat}
      </button>

      {loading && (
        <span className="shrink-0 text-xs text-muted">{text.loading ?? "…"}</span>
      )}

      {isEmpty && !loading ? null : (
        <div className="flex items-center gap-1.5">
          {sessions.map((session) => {
            const active = session.id === activeSessionId;
            const time = relativeTime(session.updatedAt ?? session.createdAt);
            const label =
              session.title ||
              `${text.sessionFallback} ${session.id.slice(-6)}`;
            return (
              <button
                key={session.id}
                type="button"
                onClick={() => onSelect(session.id)}
                title={label}
                className={cn(
                  "inline-flex max-w-[220px] shrink-0 items-center gap-1 rounded-full border px-2.5 py-1",
                  "text-xs transition-colors",
                  active
                    ? "border-transparent bg-[var(--button-bg)] text-[var(--button-fg)]"
                    : "border-divider bg-surface text-muted hover:bg-surface-2 hover:text-ink",
                )}
              >
                <ChatCircle size={12} weight="regular" className="shrink-0" aria-hidden />
                <span className="truncate">{label}</span>
                {time && (
                  <span
                    className={cn(
                      "shrink-0 font-mono text-[10px]",
                      active ? "opacity-70" : "text-muted",
                    )}
                  >
                    {time}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
