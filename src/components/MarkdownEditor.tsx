"use client";

/**
 * Productarium markdown editor for knowledge node content_md (item 2).
 *
 * Dependency-free split-pane editor: a textarea with a lightweight formatting
 * toolbar + a live preview pane rendered through the existing <Markdown />
 * component (which already renders Mermaid diagrams via <Mermaid />).
 *
 * Why not Milkdown/MDX-Editor: the npm registry is unreachable in this build
 * environment, so no new deps could be added. This component is intentionally
 * drop-in: it exposes the same `{ value, onChange, readOnly }` contract a
 * richer editor would, so swapping in @mdxeditor/editor (or Milkdown) later is
 * a one-file change.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import {
  ArrowsOutSimple,
  Code,
  CodeBlock,
  Eye,
  List,
  ListNumbers,
  MarkdownLogo,
  PencilSimple,
  Quotes,
  TextB,
  TextHOne,
  TextHTwo,
  TextItalic,
  TextStrikethrough,
} from "@phosphor-icons/react";
import dynamic from "next/dynamic";
import { Button, cn, Spinner, Textarea } from "@/components/ui";
import { useLanguage } from "@/contexts/LanguageContext";

// Mermaid touches `window`/`document` at import; load the preview renderer
// client-side only to stay SSR-safe.
const Markdown = dynamic(() => import("@/components/Markdown"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center gap-2 text-xs text-muted">
      <Spinner /> Loading preview…
    </div>
  ),
});

const TOOL_TITLE_KEYS = [
  "heading1",
  "heading2",
  "bold",
  "italic",
  "strikethrough",
  "bulletList",
  "numberedList",
  "quote",
  "inlineCode",
  "codeBlock",
  "mermaidDiagram",
] as const;

type View = "edit" | "split" | "preview";

interface MarkdownEditorProps {
  value: string;
  onChange: (next: string) => void;
  readOnly?: boolean;
  placeholder?: string;
  minHeight?: number;
  className?: string;
}

interface InsertSpec {
  /** Wrap selection: prefix + selection + suffix. */
  prefix?: string;
  suffix?: string;
  /** Insert a block prefix at line start (e.g. "# "). */
  linePrefix?: string;
  /** Insert raw text at cursor (e.g. a mermaid block). */
  block?: string;
  /** Placeholder used when there is no selection. */
  placeholder?: string;
}

export function MarkdownEditor({
  value,
  onChange,
  readOnly = false,
  placeholder,
  minHeight = 320,
  className,
}: MarkdownEditorProps) {
  const ref = useRef<HTMLTextAreaElement | null>(null);
  const [view, setView] = useState<View>("split");
  const { messages } = useLanguage();
  const t = messages?.markdownEditor ?? {};
  const resolvedPlaceholder = placeholder ?? t.placeholder ?? "Write markdown…";

  const apply = useCallback(
    (spec: InsertSpec) => {
      const el = ref.current;
      if (!el || readOnly) return;
      const start = el.selectionStart ?? value.length;
      const end = el.selectionEnd ?? value.length;
      const sel = value.slice(start, end);
      const selText = sel || spec.placeholder || "";

      if (spec.block) {
        const block = spec.block;
        const next = value.slice(0, start) + block + value.slice(end);
        onChange(next);
        queueMicrotask(() => {
          el.focus();
          const pos = start + block.length;
          el.setSelectionRange(pos, pos);
        });
        return;
      }

      if (spec.linePrefix) {
        const linePrefix = spec.linePrefix;
        const lineStart = value.lastIndexOf("\n", start - 1) + 1;
        const next =
          value.slice(0, lineStart) + linePrefix + value.slice(lineStart);
        onChange(next);
        queueMicrotask(() => {
          el.focus();
          const pos = start + linePrefix.length;
          el.setSelectionRange(pos, pos);
        });
        return;
      }

      const prefix = spec.prefix ?? "";
      const suffix = spec.suffix ?? "";
      const inserted = prefix + selText + suffix;
      const next = value.slice(0, start) + inserted + value.slice(end);
      onChange(next);
      queueMicrotask(() => {
        el.focus();
        const s = start + prefix.length;
        el.setSelectionRange(s, s + selText.length);
      });
    },
    [onChange, readOnly, value],
  );

  const tools = useMemo(
    () => [
      { icon: TextHOne, spec: { linePrefix: "# " }, titleKey: TOOL_TITLE_KEYS[0] },
      { icon: TextHTwo, spec: { linePrefix: "## " }, titleKey: TOOL_TITLE_KEYS[1] },
      { icon: TextB, spec: { prefix: "**", suffix: "**", placeholder: "bold" }, titleKey: TOOL_TITLE_KEYS[2] },
      { icon: TextItalic, spec: { prefix: "*", suffix: "*", placeholder: "italic" }, titleKey: TOOL_TITLE_KEYS[3] },
      { icon: TextStrikethrough, spec: { prefix: "~~", suffix: "~~", placeholder: "strike" }, titleKey: TOOL_TITLE_KEYS[4] },
      { icon: List, spec: { linePrefix: "- " }, titleKey: TOOL_TITLE_KEYS[5] },
      { icon: ListNumbers, spec: { linePrefix: "1. " }, titleKey: TOOL_TITLE_KEYS[6] },
      { icon: Quotes, spec: { linePrefix: "> " }, titleKey: TOOL_TITLE_KEYS[7] },
      { icon: Code, spec: { prefix: "`", suffix: "`", placeholder: "code" }, titleKey: TOOL_TITLE_KEYS[8] },
      {
        icon: CodeBlock,
        spec: { block: "\n```\ncode\n```\n" },
        titleKey: TOOL_TITLE_KEYS[9],
      },
      {
        icon: MarkdownLogo,
        spec: {
          block: "\n```mermaid\nflowchart LR\n  A --> B\n```\n",
        },
        titleKey: TOOL_TITLE_KEYS[10],
      },
    ],
    [],
  );

  const viewBtns: { key: View; labelKey: "edit" | "split" | "preview"; icon: typeof Eye }[] = [
    { key: "edit", labelKey: "edit", icon: PencilSimple },
    { key: "split", labelKey: "split", icon: ArrowsOutSimple },
    { key: "preview", labelKey: "preview", icon: Eye },
  ];

  return (
    <div className={cn("rounded-md border border-divider bg-surface", className)}>
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-1 border-b border-divider bg-surface-2 px-2 py-1.5">
        <div className="flex flex-wrap items-center gap-0.5">
          {tools.map((tool) => {
            const title = t?.[tool.titleKey] ?? tool.titleKey;
            return (
            <button
              key={tool.titleKey}
              type="button"
              title={title}
              aria-label={title}
              disabled={readOnly}
              onClick={() => apply(tool.spec)}
              className="inline-flex h-7 w-7 items-center justify-center rounded text-muted transition-colors hover:bg-surface hover:text-ink disabled:opacity-40"
            >
              <tool.icon size={15} weight="regular" />
            </button>
            );
          })}
        </div>
        <div className="ml-auto flex items-center gap-0.5">
          {viewBtns.map((v) => (
            <button
              key={v.key}
              type="button"
              onClick={() => setView(v.key)}
              className={cn(
                "inline-flex h-7 items-center gap-1 rounded px-2 text-xs font-medium transition-colors",
                view === v.key
                  ? "bg-surface text-ink"
                  : "text-muted hover:bg-surface hover:text-ink",
              )}
            >
              <v.icon size={13} weight="regular" />
              <span className="hidden sm:inline">{t?.[v.labelKey] ?? v.labelKey}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Body */}
      <div className="grid md:grid-cols-2">
        {(view === "edit" || view === "split") && (
          <Textarea
            ref={ref}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={resolvedPlaceholder}
            readOnly={readOnly}
            rows={Math.max(12, Math.round(minHeight / 20))}
            className={cn(
              "min-h-[320px] w-full resize-y rounded-none border-0 border-divider focus:border-0 focus:outline-none",
              view === "split" && "md:border-r",
            )}
            style={{ minHeight }}
          />
        )}
        {(view === "preview" || view === "split") && (
          <div
            className="prose-editor min-h-[320px] overflow-auto px-4 py-3"
            style={{ minHeight }}
          >
            {value.trim() ? (
              <Markdown content={value} />
            ) : (
              <p className="text-sm text-muted">{t.nothingToPreview ?? "Nothing to preview yet."}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default MarkdownEditor;

// Re-export for callers that want the editor wrapped in a save bar.
export function EditorSaveBar({
  saving,
  dirty,
  onSave,
  onCancel,
  saveLabel,
}: {
  saving?: boolean;
  dirty?: boolean;
  onSave: () => void;
  onCancel?: () => void;
  saveLabel?: string;
}) {
  const { messages } = useLanguage();
  const t = messages?.markdownEditor ?? {};
  return (
    <div className="flex items-center justify-end gap-2">
      {onCancel && (
        <Button type="button" variant="ghost" onClick={onCancel} disabled={saving}>
          {t.cancel ?? "Cancel"}
        </Button>
      )}
      <Button type="button" onClick={onSave} disabled={saving || !dirty}>
        {saving ? <Spinner /> : <PencilSimple size={14} weight="regular" />}
        {saving ? (t.saving ?? "Saving…") : (saveLabel ?? t.save ?? "Save")}
      </Button>
    </div>
  );
}
