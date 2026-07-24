"use client";

/**
 * SpecViewer — a dependency-free viewer for OpenAPI / AsyncAPI specs.
 *
 * Parses JSON specs client-side and renders a structured, collapsible view
 * (info / servers / operations / schemas) inspired by better-openapi-viewer.
 * YAML specs (no YAML parser is bundled) fall back to a styled raw block so
 * the content is still readable; switching the artifact to JSON enables the
 * full structured rendering. No external libraries are used.
 */

import { useMemo, useState } from "react";
import {
  CaretDown,
  CaretRight,
  FileText,
} from "@phosphor-icons/react";
import { Tag, cn } from "@/components/ui";

type HttpMethod = string;

interface ParsedSpec {
  title: string;
  version?: string;
  description?: string;
  kind: "openapi" | "asyncapi" | "unknown";
  servers: { url: string; description?: string }[];
  operations: {
    id: string;
    method: HttpMethod;
    path: string;
    summary?: string;
    description?: string;
    operationId?: string;
    deprecated?: boolean;
  }[];
  channels?: {
    id: string;
    name: string;
    summary?: string;
    description?: string;
  }[];
  schemas: {
    name: string;
    description?: string;
    type?: string;
    raw: unknown;
  }[];
}

function asString(v: unknown): string | undefined {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return undefined;
}

function parseSpec(content: string, kindHint?: string): ParsedSpec | null {
  const text = content.trim();
  if (!text) return null;

  let doc: Record<string, unknown>;
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      doc = parsed as Record<string, unknown>;
    } else {
      return null;
    }
  } catch {
    // Not JSON — likely YAML; caller renders the raw fallback.
    return null;
  }

  const kind: ParsedSpec["kind"] =
    "asyncapi" in doc || kindHint === "asyncapi"
      ? "asyncapi"
      : "openapi" in doc || kindHint === "openapi"
        ? "openapi"
        : "unknown";

  const info = (doc.info ?? null) as Record<string, unknown> | null;
  const title = asString(info?.title) ?? "Untitled spec";
  const version = asString(info?.version);
  const description = asString(info?.description);

  const servers: ParsedSpec["servers"] = [];
  const serversField = doc.servers;
  if (serversField && typeof serversField === "object") {
    const entries = Array.isArray(serversField)
      ? (serversField as unknown[])
      : Object.values(serversField as Record<string, unknown>);
    for (const s of entries) {
      if (s && typeof s === "object") {
        const so = s as Record<string, unknown>;
        const url = asString(so.url) ?? asString(so.baseUrl);
        if (url) servers.push({ url, description: asString(so.description) });
      }
    }
  }

  const operations: ParsedSpec["operations"] = [];
  const channels: ParsedSpec["channels"] = [];

  // OpenAPI: paths -> { path: { method: operation } }
  const paths = doc.paths;
  if (paths && typeof paths === "object" && !Array.isArray(paths)) {
    for (const [path, opsRaw] of Object.entries(paths as Record<string, unknown>)) {
      if (!opsRaw || typeof opsRaw !== "object") continue;
      const ops = opsRaw as Record<string, unknown>;
      for (const [method, opRaw] of Object.entries(ops)) {
        const m = method.toLowerCase();
        if (!/^(get|post|put|patch|delete|head|options|trace)$/.test(m)) continue;
        if (!opRaw || typeof opRaw !== "object") continue;
        const op = opRaw as Record<string, unknown>;
        operations.push({
          id: `${m.toUpperCase()} ${path}`,
          method: m.toUpperCase(),
          path,
          summary: asString(op.summary),
          description: asString(op.description),
          operationId: asString(op.operationId),
          deprecated: Boolean(op.deprecated),
        });
      }
    }
  }

  // AsyncAPI: channels -> { name: { publish/subscribe } }
  const channelsField = doc.channels;
  if (channelsField && typeof channelsField === "object" && !Array.isArray(channelsField)) {
    for (const [name, chRaw] of Object.entries(channelsField as Record<string, unknown>)) {
      if (!chRaw || typeof chRaw !== "object") continue;
      const ch = chRaw as Record<string, unknown>;
      channels.push({
        id: name,
        name,
        summary: asString(ch.summary),
        description: asString(ch.description),
      });
    }
  }

  // Schemas: components.schemas (OpenAPI) or components.schemas (AsyncAPI)
  const schemas: ParsedSpec["schemas"] = [];
  const components = doc.components as Record<string, unknown> | undefined;
  const schemasField = components?.schemas;
  if (schemasField && typeof schemasField === "object" && !Array.isArray(schemasField)) {
    for (const [name, schemaRaw] of Object.entries(schemasField as Record<string, unknown>)) {
      if (!schemaRaw || typeof schemaRaw !== "object") continue;
      const schema = schemaRaw as Record<string, unknown>;
      schemas.push({
        name,
        description: asString(schema.description),
        type: asString(schema.type),
        raw: schemaRaw,
      });
    }
  }

  return { title, version, description, kind, servers, operations, channels, schemas };
}

function Collapsible({
  title,
  count,
  defaultOpen = false,
  children,
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-md border border-divider bg-surface">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-4 py-3 text-left"
      >
        {open ? (
          <CaretDown size={14} weight="bold" className="text-muted" />
        ) : (
          <CaretRight size={14} weight="bold" className="text-muted" />
        )}
        <span className="text-sm font-medium text-ink">{title}</span>
        {typeof count === "number" && (
          <span className="text-xs text-muted">{count}</span>
        )}
      </button>
      {open && <div className="border-t border-divider px-4 py-3">{children}</div>}
    </div>
  );
}

const METHOD_TONE: Record<string, string> = {
  GET: "bg-tag-blue-bg text-tag-blue-fg",
  POST: "bg-tag-green-bg text-tag-green-fg",
  PUT: "bg-tag-yellow-bg text-tag-yellow-fg",
  PATCH: "bg-tag-yellow-bg text-tag-yellow-fg",
  DELETE: "bg-tag-red-bg text-tag-red-fg",
  HEAD: "bg-tag-neutral-bg text-tag-neutral-fg",
  OPTIONS: "bg-tag-neutral-bg text-tag-neutral-fg",
};

function jsonPreview(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function SpecViewer({
  content,
  kind,
}: {
  content: string;
  kind?: string;
}) {
  const parsed = useMemo(() => parseSpec(content, kind), [content, kind]);

  if (!parsed) {
    // Not JSON (likely YAML) or empty: render a friendly raw fallback.
    return (
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-sm text-muted">
          <FileText size={16} weight="regular" />
          <span>
            {content.trim()
              ? "Structured view is available for JSON specs. This spec looks like YAML — showing the raw source."
              : "No specification content."}
          </span>
        </div>
        {content.trim() && (
          <pre className="max-h-[60vh] overflow-auto rounded-md border border-divider bg-surface-2 p-4 font-mono text-xs leading-relaxed text-ink">
            {content}
          </pre>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Info */}
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="font-editorial text-2xl tracking-tight text-ink">
            {parsed.title}
          </h2>
          {parsed.version && (
            <Tag tone="neutral">v{parsed.version}</Tag>
          )}
          {parsed.kind !== "unknown" && (
            <Tag tone="green">{parsed.kind}</Tag>
          )}
        </div>
        {parsed.description && (
          <p className="mt-2 max-w-3xl text-sm text-muted">
            {parsed.description}
          </p>
        )}
      </div>

      {/* Servers */}
      {parsed.servers.length > 0 && (
        <Collapsible
          title="Servers"
          count={parsed.servers.length}
          defaultOpen={parsed.servers.length <= 3}
        >
          <ul className="flex flex-col gap-2">
            {parsed.servers.map((s, i) => (
              <li
                key={i}
                className="flex flex-wrap items-center gap-2 text-sm"
              >
                <code className="rounded bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink">
                  {s.url}
                </code>
                {s.description && (
                  <span className="text-muted">{s.description}</span>
                )}
              </li>
            ))}
          </ul>
        </Collapsible>
      )}

      {/* Operations (OpenAPI) */}
      {parsed.operations.length > 0 && (
        <Collapsible
          title="Operations"
          count={parsed.operations.length}
          defaultOpen
        >
          <ul className="flex flex-col gap-1">
            {parsed.operations.map((op) => (
              <li
                key={op.id}
                className="flex items-start gap-3 rounded-md px-2 py-2 hover:bg-surface-2"
              >
                <span
                  className={cn(
                    "inline-flex w-16 shrink-0 justify-center rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold uppercase",
                    METHOD_TONE[op.method] ?? "bg-tag-neutral-bg text-tag-neutral-fg",
                  )}
                >
                  {op.method}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <code className="font-mono text-xs text-ink">{op.path}</code>
                    {op.deprecated && (
                      <Tag tone="red">deprecated</Tag>
                    )}
                  </div>
                  {op.summary && (
                    <p className="mt-0.5 text-sm text-muted">{op.summary}</p>
                  )}
                  {op.description && op.description !== op.summary && (
                    <p className="mt-0.5 text-xs text-muted">{op.description}</p>
                  )}
                </div>
                {op.operationId && (
                  <code className="hidden shrink-0 font-mono text-[11px] text-muted md:block">
                    {op.operationId}
                  </code>
                )}
              </li>
            ))}
          </ul>
        </Collapsible>
      )}

      {/* Channels (AsyncAPI) */}
      {parsed.channels && parsed.channels.length > 0 && (
        <Collapsible
          title="Channels"
          count={parsed.channels.length}
          defaultOpen
        >
          <ul className="flex flex-col gap-1">
            {parsed.channels.map((ch) => (
              <li
                key={ch.id}
                className="rounded-md px-2 py-2 hover:bg-surface-2"
              >
                <code className="font-mono text-xs text-ink">{ch.name}</code>
                {ch.summary && (
                  <p className="mt-0.5 text-sm text-muted">{ch.summary}</p>
                )}
                {ch.description && (
                  <p className="mt-0.5 text-xs text-muted">{ch.description}</p>
                )}
              </li>
            ))}
          </ul>
        </Collapsible>
      )}

      {/* Schemas */}
      {parsed.schemas.length > 0 && (
        <Collapsible
          title="Schemas"
          count={parsed.schemas.length}
          defaultOpen={parsed.schemas.length <= 5}
        >
          <div className="flex flex-col gap-2">
            {parsed.schemas.map((s) => (
              <div
                key={s.name}
                className="rounded-md border border-divider bg-surface-2 p-3"
              >
                <div className="flex items-center gap-2">
                  <code className="font-mono text-sm text-ink">{s.name}</code>
                  {s.type && (
                    <Tag tone="neutral">{s.type}</Tag>
                  )}
                </div>
                {s.description && (
                  <p className="mt-1 text-xs text-muted">{s.description}</p>
                )}
                <pre className="mt-2 max-h-64 overflow-auto rounded bg-surface p-3 font-mono text-[11px] leading-relaxed text-muted">
                  {jsonPreview(s.raw)}
                </pre>
              </div>
            ))}
          </div>
        </Collapsible>
      )}

      {parsed.operations.length === 0 &&
        (!parsed.channels || parsed.channels.length === 0) &&
        parsed.servers.length === 0 &&
        parsed.schemas.length === 0 && (
          <p className="text-sm text-muted">
            The spec parsed but contained no servers, operations, channels, or schemas.
      </p>
        )}
    </div>
  );
}
