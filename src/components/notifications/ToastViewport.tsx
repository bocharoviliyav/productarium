"use client";

/**
 * Floating toast stack (top-right). Renders only the active (non-hidden)
 * notifications. Each toast auto-hides via the NotificationContext timer; the
 * tray retains the full session history.
 */

import { useEffect, useState } from "react";
import { X } from "@phosphor-icons/react";
import { useNotifications } from "@/contexts/NotificationContext";
import { cn } from "@/components/ui";
import { TONE_STYLES } from "./toneStyles";

export function ToastViewport() {
  const { notifications, dismiss } = useNotifications();
  // Avoid SSR/CSR mismatch: render nothing until mounted on the client.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const active = mounted ? notifications.filter((n) => !n.hidden) : [];

  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      className="pointer-events-none fixed right-4 top-16 z-[60] flex w-[min(92vw,360px)] flex-col gap-2"
    >
      {active.map((n) => {
        const tone = TONE_STYLES[n.tone];
        return (
          <div
            key={n.id}
            role={n.tone === "error" ? "alert" : "status"}
            className={cn(
              "toast-enter pointer-events-auto flex items-start gap-3 rounded-lg border bg-surface px-4 py-3 shadow-[0_8px_40px_rgba(0,0,0,0.12)]",
              tone.border,
            )}
          >
            <span
              className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", tone.dot)}
              aria-hidden
            />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-ink">{n.title}</p>
              {n.message && (
                <p className="mt-0.5 text-[13px] leading-relaxed text-muted">
                  {n.message}
                </p>
              )}
            </div>
            <button
              type="button"
              onClick={() => dismiss(n.id)}
              className="shrink-0 rounded p-0.5 text-muted opacity-60 transition-opacity hover:opacity-100"
              aria-label="Dismiss notification"
            >
              <X size={14} weight="bold" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

export default ToastViewport;
