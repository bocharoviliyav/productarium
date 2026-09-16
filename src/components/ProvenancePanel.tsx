"use client";

import dynamic from "next/dynamic";
import { SealCheck, SealQuestion, SealWarning } from "@phosphor-icons/react";
import { Tag } from "@/components/ui";
import { useLanguage } from "@/contexts/LanguageContext";
import type { PageProvenance } from "@/lib/types";

const Markdown = dynamic(() => import("@/components/Markdown"), {
  ssr: false,
});

/**
 * Compact "Verification" panel for the active page's provenance block
 * (api/docgen/verification.py): judge verdict + issues, citation/secret/
 * mermaid/corroborate counters, generation meta, and the extracted
 * provenance report (assumptions/gaps/confidence). Renders below the page
 * text; nothing for legacy pages without provenance.
 */
export function ProvenancePanel({ provenance }: { provenance?: PageProvenance }) {
  const { messages, fmt } = useLanguage();
  const t = messages?.artifact?.provenance ?? {};
  if (!provenance) return null;

  const verdict = provenance.judge?.verdict;
  const judgeView =
    verdict === "consistent"
      ? { tone: "green" as const, icon: SealCheck, weight: "fill" as const, label: t.judgeConsistent ?? "Judge: consistent" }
      : verdict === "inconsistent"
        ? { tone: "red" as const, icon: SealWarning, weight: "fill" as const, label: t.judgeInconsistent ?? "Judge: inconsistencies" }
        : verdict === "skipped"
          ? { tone: "neutral" as const, icon: SealQuestion, weight: "regular" as const, label: t.judgeSkipped ?? "Judge: skipped" }
          : null;
  const JudgeIcon = judgeView?.icon;

  const c = provenance.citations;
  const counts: string[] = [];
  if (c?.resolved?.length || c?.unresolved?.length || c?.removed?.length) {
    const parts = [
      c?.resolved?.length ? fmt(t.citationsResolved ?? "resolved {n}", { n: c.resolved.length }) : "",
      c?.unresolved?.length ? fmt(t.citationsUnresolved ?? "unresolved {n}", { n: c.unresolved.length }) : "",
      c?.removed?.length ? fmt(t.citationsRemoved ?? "removed {n}", { n: c.removed.length }) : "",
    ].filter(Boolean);
    counts.push(`${t.citations ?? "Citations"}: ${parts.join(", ")}`);
  }
  const m = provenance.mermaid;
  if (m) {
    const parts = [
      m.verified ? fmt(t.mermaidOk ?? "ok {n}", { n: m.verified }) : "",
      m.fixed ? fmt(t.mermaidFixed ?? "repaired {n}", { n: m.fixed }) : "",
      m.failed ? fmt(t.mermaidFailed ?? "failed {n}", { n: m.failed }) : "",
    ].filter(Boolean);
    if (parts.length) counts.push(`Mermaid: ${parts.join(", ")}`);
  }
  if (provenance.secrets_masked)
    counts.push(fmt(t.secretsMasked ?? "secrets masked: {n}", { n: provenance.secrets_masked }));
  const ungrounded = provenance.corroborate?.removed?.length ?? 0;
  if (ungrounded)
    counts.push(fmt(t.corroborateRemoved ?? "ungrounded dropped: {n}", { n: ungrounded }));

  const date = provenance.generated_at ? new Date(provenance.generated_at) : null;
  const meta = [
    provenance.model || provenance.generator,
    date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : undefined,
    provenance.regen,
  ]
    .filter(Boolean)
    .join(" · ");

  const issues = provenance.judge?.issues ?? [];
  const report =
    typeof provenance.report === "string" ? provenance.report.trim() : "";

  return (
    <div className="rounded-xl border border-divider bg-surface px-4 py-3 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium uppercase tracking-wide text-muted">
          {t.title ?? "Verification"}
        </span>
        {judgeView && JudgeIcon && (
          <Tag tone={judgeView.tone} className="gap-1">
            <JudgeIcon size={12} weight={judgeView.weight} />
            {judgeView.label}
          </Tag>
        )}
        {meta && <span className="ml-auto text-muted">{meta}</span>}
      </div>
      {counts.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-muted">
          {counts.map((s) => (
            <span key={s}>{s}</span>
          ))}
        </div>
      )}
      {issues.length > 0 && (
        <div className="mt-2">
          <p className="font-medium text-ink">{t.judgeIssues ?? "Judge issues"}</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-muted">
            {issues.map((issue, i) => (
              <li key={`${i}-${issue}`}>{issue}</li>
            ))}
          </ul>
        </div>
      )}
      {report && (
        <div className="mt-2 border-t border-divider pt-2">
          <Markdown content={report} />
        </div>
      )}
    </div>
  );
}

export default ProvenancePanel;
