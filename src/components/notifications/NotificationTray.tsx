"use client";

/**
 * Notification tray (TopBar right slot). A bell button with an unread badge;
 * click opens a dropdown listing the session's notifications (newest first)
 * with per-item dismiss and a "Clear all" action.
 */

import { useEffect, useRef, useState } from "react";
import { Bell, Trash, X } from "@phosphor-icons/react";
import { useNotifications } from "@/contexts/NotificationContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { cn } from "@/components/ui";
import { TONE_STYLES, relativeTime } from "./toneStyles";

export function NotificationTray() {
  const { notifications, unreadCount, dismiss, clearAll, markAllSeen } =
    useNotifications();
  const { messages } = useLanguage();
  const t = messages?.notifications ?? {};
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const ref = useRef<HTMLDivElement | null>(null);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  // Mark as seen once the tray is opened.
  useEffect(() => {
    if (open && unreadCount > 0) {
      markAllSeen();
    }
  }, [open, unreadCount, markAllSeen]);

  // Refresh relative timestamps while open.
  useEffect(() => {
    if (!open) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="relative inline-flex h-9 w-9 items-center justify-center rounded-md border border-divider bg-surface text-ink transition-colors hover:bg-surface-2 active:scale-[0.98]"
        aria-label={t.title || "Notifications"}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <Bell size={16} weight="regular" />
        {unreadCount > 0 && (
          <span className="absolute -right-1 -top-1 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-tag-red-fg px-1 text-[10px] font-semibold leading-none text-surface">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-11 z-50 flex max-h-[28rem] w-80 flex-col overflow-hidden rounded-md border border-divider bg-surface py-1 shadow-[0_8px_40px_rgba(0,0,0,0.12)]"
        >
          <div className="flex items-center justify-between border-b border-divider px-3 py-2">
            <span className="text-xs font-medium uppercase tracking-wide text-muted">
              {t.title || "Notifications"}
            </span>
            {notifications.length > 0 && (
              <button
                type="button"
                onClick={clearAll}
                className="inline-flex items-center gap-1 text-xs text-muted transition-colors hover:text-ink"
              >
                <Trash size={12} weight="regular" />
                {t.clearAll || "Clear all"}
              </button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="px-4 py-8 text-center text-xs text-muted">
                {t.empty || "No notifications"}
              </div>
            ) : (
              notifications.map((n) => {
                const tone = TONE_STYLES[n.tone];
                return (
                  <div
                    key={n.id}
                    className="group flex items-start gap-2.5 border-b border-divider px-3 py-2.5 last:border-b-0 hover:bg-surface-2"
                  >
                    <span
                      className={cn(
                        "mt-1.5 h-2 w-2 shrink-0 rounded-full",
                        tone.dot,
                      )}
                      aria-hidden
                    />
                    <div className="min-w-0 flex-1">
                      <p className="text-[13px] font-medium text-ink">
                        {n.title}
                      </p>
                      {n.message && (
                        <p className="mt-0.5 text-xs leading-relaxed text-muted">
                          {n.message}
                        </p>
                      )}
                      <p className="mt-1 text-[10px] uppercase tracking-wide text-muted">
                        {relativeTime(n.createdAt, now, {
                          justNow: t.justNow || "just now",
                          minutesAgo: t.minutesAgo || "{n}m ago",
                          hoursAgo: t.hoursAgo || "{n}h ago",
                        })}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => dismiss(n.id)}
                      className="shrink-0 rounded p-0.5 text-muted opacity-0 transition-opacity hover:text-ink group-hover:opacity-100"
                      aria-label="Dismiss notification"
                    >
                      <X size={12} weight="bold" />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default NotificationTray;
