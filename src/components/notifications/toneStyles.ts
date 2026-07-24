"use client";

/**
 * Tone -> token classes shared by toasts and the notification tray.
 * Rides on the existing desaturated pastel tag palette.
 */
import type { NotificationTone } from "@/contexts/NotificationContext";

export const TONE_STYLES: Record<
  NotificationTone,
  { border: string; bg: string; text: string; dot: string }
> = {
  info: {
    border: "border-tag-blue-bg",
    bg: "bg-tag-blue-bg",
    text: "text-tag-blue-fg",
    dot: "bg-tag-blue-fg",
  },
  success: {
    border: "border-tag-green-bg",
    bg: "bg-tag-green-bg",
    text: "text-tag-green-fg",
    dot: "bg-tag-green-fg",
  },
  warning: {
    border: "border-tag-yellow-bg",
    bg: "bg-tag-yellow-bg",
    text: "text-tag-yellow-fg",
    dot: "bg-tag-yellow-fg",
  },
  error: {
    border: "border-tag-red-bg",
    bg: "bg-tag-red-bg",
    text: "text-tag-red-fg",
    dot: "bg-tag-red-fg",
  },
};

/** Compact relative-time label (e.g. "just now", "3m ago", "2h ago"). */
export function relativeTime(
  createdAt: number,
  now: number,
  t: { justNow: string; minutesAgo: string; hoursAgo: string },
): string {
  const diffMs = Math.max(0, now - createdAt);
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return t.justNow;
  if (mins < 60) return t.minutesAgo.replace("{n}", String(mins));
  const hours = Math.floor(mins / 60);
  return t.hoursAgo.replace("{n}", String(hours));
}
