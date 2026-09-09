"use client";

/**
 * Shared MCP status dot (ok / error / unknown), Zed-style.
 *
 * Warm-monochrome friendly: solid dots reuse the tag foreground tokens so
 * they adapt to the light/dark themes. The tooltip surfaces `status_error`
 * from the backend (e.g. the last health-check failure reason).
 */

import { cn } from "@/components/ui";
import type { McpServerStatus } from "@/lib/types";

const DOT_CLASS: Record<McpServerStatus, string> = {
  ok: "bg-tag-green-fg",
  error: "bg-tag-red-fg",
  unknown: "bg-tag-neutral-fg",
};

export function McpStatusDot({
  status,
  title,
  className,
}: {
  status: McpServerStatus;
  /** Tooltip text — pass the localized status label and/or `status_error`. */
  title?: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-block h-2 w-2 shrink-0 rounded-full",
        DOT_CLASS[status],
        className,
      )}
      title={title}
      aria-hidden
    />
  );
}
