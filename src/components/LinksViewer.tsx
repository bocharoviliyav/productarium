"use client";

/**
 * LinksViewer — a read-only table view for a `links` artifact.
 *
 * Renders the curated link list parsed from `artifact.content` (JSON list of
 * `{url, description?}`, or the legacy free-form Markdown). External links open
 * in a new tab. Empty content shows a quiet placeholder.
 */

import { ArrowSquareOut, LinkSimple } from "@phosphor-icons/react";
import { parseLinksContent } from "@/lib/types";

export function LinksViewer({ content }: { content: string }) {
  const items = parseLinksContent(content);

  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-10 text-center">
        <span className="flex h-9 w-9 items-center justify-center rounded-full bg-surface-2 text-muted">
          <LinkSimple size={18} weight="regular" />
        </span>
        <p className="text-sm text-muted">No links yet.</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-md border border-divider">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
            <th className="w-1/2 px-4 py-2.5 font-medium">Link</th>
            <th className="px-4 py-2.5 font-medium">Description</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => {
            const url = item.url.trim();
            const desc = (item.description ?? "").trim();
            const isUrl = /^https?:\/\//i.test(url);
            return (
              <tr
                key={idx}
                className="border-t border-divider align-top hover:bg-surface-2"
              >
                <td className="px-4 py-3">
                  {url ? (
                    isUrl ? (
                      <a
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 font-mono text-xs text-ink underline-offset-2 hover:underline"
                      >
                        {url}
                        <ArrowSquareOut
                          size={12}
                          weight="regular"
                          className="text-muted"
                        />
                      </a>
                    ) : (
                      <span className="font-mono text-xs text-ink">{url}</span>
                    )
                  ) : (
                    <span className="text-xs text-muted">—</span>
                  )}
                </td>
                <td className="px-4 py-3 text-muted">
                  {desc || <span className="text-xs text-muted">—</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
