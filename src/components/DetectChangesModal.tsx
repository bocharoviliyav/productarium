"use client";

/**
 * DetectChangesModal — SPEC-010 detect summary: what changed in the source,
 * which pages are impacted, and the confirmation to run the incremental
 * update (empty impact / spec → full regeneration, labelled accordingly).
 * Labels come from the shared `artifact` i18n section.
 */

import { Lightning } from "@phosphor-icons/react";
import { useLanguage } from "@/contexts/LanguageContext";
import { Button, Modal, Tag } from "@/components/ui";
import type { DetectResult } from "@/lib/types";

const LIST_CAP = 30;

function DetectList({ label, items }: { label: string; items: string[] }) {
  if (!items.length) return null;
  const shown = items.slice(0, LIST_CAP);
  return (
    <div>
      <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
        {label} ({items.length})
      </p>
      <ul className="max-h-36 overflow-auto rounded-md border border-divider bg-surface-2 p-3 font-mono text-xs leading-relaxed text-muted">
        {shown.map((it) => (
          <li key={it} className="truncate">{it}</li>
        ))}
        {items.length > shown.length && <li>+{items.length - shown.length}</li>}
      </ul>
    </div>
  );
}

export function DetectChangesModal({
  result,
  onClose,
  onUpdate,
}: {
  result: DetectResult | null;
  onClose: () => void;
  onUpdate: () => void;
}) {
  const { messages } = useLanguage();
  const t = messages?.artifact ?? {};
  return (
    <Modal
      open={result !== null}
      onClose={onClose}
      title={t.detectTitle ?? "Change check"}
      footer={null}
    >
      {result && (
        <div className="flex flex-col gap-4">
          <p className="text-sm text-muted">{result.reason}</p>
          <DetectList
            label={t.detectChangedFiles ?? "Changed files"}
            items={result.changed_files ?? []}
          />
          <DetectList
            label={t.detectChangedObjects ?? "Changed objects"}
            items={result.changed_objects ?? []}
          />
          {result.impacted_pages.length > 0 ? (
            <div>
              <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
                {t.detectImpactedPages ?? "Impacted pages"} ({result.impacted_pages.length})
              </p>
              <ul className="max-h-44 overflow-auto rounded-md border border-divider bg-surface-2 p-3 text-sm">
                {result.impacted_pages.map((p) => (
                  <li key={p.id} className="flex items-center justify-between gap-2 py-0.5">
                    <span className="min-w-0 truncate text-ink">{p.title}</span>
                    {p.new ? <Tag tone="blue">{t.detectNewPage ?? "new"}</Tag> : null}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-sm text-muted">
              {t.detectFullRegen ?? "Documentation will be regenerated in full."}
            </p>
          )}
          <div className="flex items-center justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              {t.detectLater ?? "Later"}
            </Button>
            <Button onClick={onUpdate}>
              <Lightning size={16} weight="fill" />
              {result.impacted_pages.length > 0
                ? (t.updateChanged ?? "Update changed only")
                : (t.updateAll ?? "Update documentation")}
            </Button>
          </div>
        </div>
      )}
    </Modal>
  );
}
