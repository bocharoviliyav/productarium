"use client";

import { useState } from "react";
import { Article, ArrowClockwise, Lightning, Spinner } from "@phosphor-icons/react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useNotifications } from "@/contexts/NotificationContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { Button, Card, cn } from "@/components/ui";
import type { Product } from "@/lib/types";

const Markdown = dynamic(() => import("@/components/Markdown"), {
  ssr: false,
  loading: () => <div className="text-xs text-muted">Rendering…</div>,
});

/**
 * AI summary block (item 4). Renders `product.summary` as markdown, or an
 * empty state with a "Generate summary" button that POSTs
 * /api/products/{id}/summary (expert agent / LLM over the product's artifacts)
 * and then calls `onRefresh` so the parent refetches the product.
 */
export function SummaryBlock({
  product,
  onRefresh,
  className,
}: {
  product: Product;
  onRefresh?: () => void | Promise<void>;
  className?: string;
}) {
  const [busy, setBusy] = useState(false);
  const { notify } = useNotifications();
  const router = useRouter();
  const { messages } = useLanguage();
  const t = messages?.summary ?? {};

  const generate = async () => {
    setBusy(true);
    try {
      const res = await fetch(`/api/products/${product.id}/summary`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (res.status === 401) {
        router.replace(`/login?next=/products/${product.id}`);
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Summary generation failed (${res.status})`);
      }
      await onRefresh?.();
    } catch (e) {
      notify({
        tone: "error",
        title: t.failedTitle ?? "Summary generation failed",
        message: e instanceof Error ? e.message : (t.failedMessage ?? "Summary generation failed"),
      });
    } finally {
      setBusy(false);
    }
  };

  const hasSummary = Boolean(product.summary && product.summary.trim());

  return (
    <Card className={cn("p-6", className)}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-surface-2 text-ink">
            <Article size={16} weight="regular" />
          </span>
          <div>
            <h2 className="font-editorial text-base tracking-tight text-ink">
              {t.title ?? "Summary"}
            </h2>
            <p className="text-xs text-muted">
              {t.subtitle ?? ""}
            </p>
          </div>
        </div>
        {hasSummary ? (
          <Button size="sm" variant="subtle" onClick={generate} disabled={busy}>
            {busy ? <Spinner /> : <ArrowClockwise size={14} weight="bold" />}
            {busy ? (t.generating ?? "Generating…") : (t.regenerate ?? "Regenerate")}
          </Button>
        ) : (
          <Button size="sm" variant="subtle" onClick={generate} disabled={busy}>
            {busy ? <Spinner /> : <Lightning size={14} weight="fill" />}
            {busy ? (t.generating ?? "Generating…") : (t.generate ?? "Generate summary")}
          </Button>
        )}
      </div>

      <div className="mt-4">
        {hasSummary ? (
          <div className="prose-editor max-w-none text-sm text-ink">
            <Markdown content={product.summary!} />
          </div>
        ) : (
          <p className="text-sm text-muted">
            {t.empty ?? ""}
          </p>
        )}
      </div>
    </Card>
  );
}

export default SummaryBlock;
