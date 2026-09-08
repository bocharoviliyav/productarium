"use client";

/**
 * Compact tool activity chip for the expert agent chat (wave B).
 *
 * Renders one `tool_call` / `tool_result` pair from the SSE stream as a
 * quiet single-line row: a Wrench icon, the tool name (mono, medium), the
 * short argument rendering, and — once the result arrives — a muted summary
 * after a middle dot. Running tools show a tiny spinner instead.
 *
 * Style: minimalist-ui — 1px divider border, surface-2 fill, mono type,
 * no gradients or shadows.
 */

import { Check, Spinner, Wrench } from "@phosphor-icons/react";
import { formatToolArgs, type ToolEvent } from "@/lib/expertChat";
import { cn } from "@/components/ui";

export function ToolEventChip({ event }: { event: ToolEvent }) {
  const running = event.result === null;
  const argsText = formatToolArgs(event.args);
  const tooltip = running ? argsText : event.result || argsText;
  return (
    <div
      className={cn(
        "inline-flex max-w-full items-center gap-1.5 rounded-md border border-divider",
        "bg-surface-2 px-2 py-1 font-mono text-[11px] leading-5 text-muted",
      )}
      title={formatToolArgs(event.args, 400) === argsText ? tooltip : `${argsText}\n${tooltip}`}
    >
      <span className="shrink-0 text-muted" aria-hidden>
        <Wrench size={12} weight="regular" />
      </span>
      <span className="shrink-0 font-medium text-ink">{event.name}</span>
      {argsText ? <span className="truncate">{argsText}</span> : null}
      {running ? (
        <span className="ml-0.5 shrink-0" aria-hidden>
          <Spinner size={11} />
        </span>
      ) : (
        event.result && (
          <span className="flex min-w-0 items-center gap-1">
            <span className="shrink-0 text-muted" aria-hidden>
              <Check size={11} weight="bold" />
            </span>
            <span className="truncate">{event.result}</span>
          </span>
        )
      )}
    </div>
  );
}
