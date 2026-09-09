/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

/**
 * Product-scoped MCP Servers panel (Zed-style, wave C).
 *
 * Lists the MCP servers bound to this product (GET
 * /api/products/{product_id}/mcp) with a transport badge, a status dot
 * (tooltip carries the server's last status_error), an instant enable toggle
 * (PUT .../mcp/{binding_id} {enabled}) and a tools counter. Each row expands
 * into an allowed-tools editor (null = all tools exposed to the expert agent):
 * admins get tool-name chips discovered from the registry cache
 * (GET /api/admin/mcp/servers/{id}/tools), non-admins get a manual
 * comma-separated input. Detach = DELETE .../mcp/{binding_id}.
 *
 * The attach button is gated on the global admin registry: it is rendered for
 * admins and hidden gracefully when GET /api/admin/mcp/servers returns 403.
 */

import { useCallback, useEffect, useState } from "react";
import { CaretDown, CaretUp, Plugs, Plus, Trash } from "@phosphor-icons/react";
import { useAuth } from "@/contexts/AuthContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import {
  Button,
  Card,
  IconButton,
  Input,
  Label,
  Modal,
  SectionHeader,
  Select,
  Spinner,
  Switch,
  Tag,
  Textarea,
  cn,
} from "@/components/ui";
import { McpStatusDot } from "@/components/mcp/McpStatusDot";
import type {
  McpServer,
  McpServerBinding,
  McpToolInfo,
  McpTransport,
} from "@/lib/types";

/** Transport tag tone (shared by the binding rows and the attach picker). */
function transportTone(transport: string) {
  if (transport === "http") return "blue" as const;
  if (transport === "sse") return "green" as const;
  return "neutral" as const;
}

/** Parse `key=value` lines; returns the map plus any malformed raw lines. */
function parseKeyValueLines(text: string): {
  map: Record<string, string>;
  invalid: string[];
} {
  const map: Record<string, string> = {};
  const invalid: string[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    const key = eq > 0 ? line.slice(0, eq).trim() : "";
    if (!key) {
      invalid.push(line);
      continue;
    }
    map[key] = line.slice(eq + 1).trim();
  }
  return { map, invalid };
}

interface CreateServerFormState {
  name: string;
  transport: McpTransport;
  url: string;
  headersText: string;
  command: string;
  argsText: string;
  envText: string;
  enabled: boolean;
}

const EMPTY_CREATE_FORM: CreateServerFormState = {
  name: "",
  transport: "http",
  url: "",
  headersText: "",
  command: "",
  argsText: "",
  envText: "",
  enabled: true,
};

/* ------------------------------------------------------------------ */
/* Allowed-tools editor (chips for admins, manual input otherwise)      */
/* ------------------------------------------------------------------ */

function AllowlistEditor({
  binding,
  tools,
  saving,
  onSave,
  tm,
  fmt,
}: {
  binding: McpServerBinding;
  /** Discovered tool list, or null when unavailable (non-admin / not cached). */
  tools: McpToolInfo[] | null;
  saving: boolean;
  onSave: (allowed: string[] | null) => void;
  tm: any;
  fmt: (template: string | undefined, vars?: Record<string, string | number>) => string;
}) {
  const [draft, setDraft] = useState<string[] | null>(binding.allowed_tools);

  // Re-seed the draft when the binding identity/allowlist changes (e.g. after
  // a successful save elsewhere in the panel).
  useEffect(() => {
    setDraft(binding.allowed_tools);
  }, [binding.id, binding.allowed_tools]);

  const dirty = JSON.stringify(draft) !== JSON.stringify(binding.allowed_tools);

  const toggleTool = (name: string) => {
    setDraft((prev) => {
      if (prev === null) {
        // "All tools" → deselecting one chip switches to an explicit list of
        // the remaining tools.
        return (tools ?? []).map((t) => t.name).filter((n) => n !== name);
      }
      return prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name];
    });
  };

  if (tools === null) {
    // Manual mode (non-admin, or discovery cache unavailable): a comma-
    // separated list of tool names; empty = all tools (null).
    const text = draft === null ? "" : draft.join(", ");
    return (
      <div className="grid gap-2 pb-4">
        <p className="text-xs text-muted">{tm.allowlistManualHint ?? ""}</p>
        <div className="flex items-center gap-2">
          <Input
            value={text}
            onChange={(e) => {
              const parts = e.target.value
                .split(",")
                .map((s) => s.trim())
                .filter(Boolean);
              setDraft(parts.length ? Array.from(new Set(parts)) : null);
            }}
            placeholder={tm.allowlistManualPlaceholder ?? "tool names, comma-separated (empty = all)"}
            className="font-mono text-sm"
          />
          <Button
            size="sm"
            onClick={() => onSave(draft)}
            disabled={saving || !dirty}
          >
            {saving ? <Spinner /> : null}
            {tm.allowlistSave ?? "Save"}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 pb-4">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setDraft(null)}
          className={cn(
            "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium uppercase tracking-wide transition-colors",
            draft === null
              ? "border-transparent bg-tag-blue-bg text-tag-blue-fg"
              : "border-divider bg-surface text-muted hover:bg-surface-2",
          )}
        >
          {tm.allTools ?? "All tools"}
        </button>
        {tools.map((t) => {
          const selected = draft === null || draft.includes(t.name);
          return (
            <button
              key={t.name}
              type="button"
              title={t.description || t.name}
              onClick={() => toggleTool(t.name)}
              className={cn(
                "inline-flex max-w-[240px] items-center rounded-full border px-2.5 py-0.5 font-mono text-xs transition-colors",
                selected
                  ? "border-transparent bg-tag-green-bg text-tag-green-fg"
                  : "border-divider bg-surface text-muted line-through hover:bg-surface-2",
              )}
            >
              <span className="truncate">{t.name}</span>
            </button>
          );
        })}
        {tools.length === 0 && (
          <span className="text-xs text-muted">
            {tm.noToolsDiscovered ?? "No tools discovered yet."}
          </span>
        )}
      </div>
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-muted">
          {draft === null
            ? (tm.allowlistAllHint ?? "All discovered tools are exposed to the expert agent.")
            : fmt(tm.allowlistSelectedHint ?? "{n} of {total} tools allowed.", {
                n: draft.length,
                total: tools.length,
              })}
        </p>
        <Button size="sm" onClick={() => onSave(draft)} disabled={saving || !dirty}>
          {saving ? <Spinner /> : null}
          {tm.allowlistSave ?? "Save"}
        </Button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Panel                                                               */
/* ------------------------------------------------------------------ */

export function McpServersPanel({ productId }: { productId: string }) {
  const { user } = useAuth();
  const { notify } = useNotifications();
  const { messages, fmt } = useLanguage();
  const tm = (messages?.product?.mcp ?? {}) as any;

  const isAdmin = user?.role === "admin";

  const [bindings, setBindings] = useState<McpServerBinding[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [savingId, setSavingId] = useState<string | null>(null);

  // Admin registry state (attach picker + status_error tooltips + tools cache).
  const [registryDenied, setRegistryDenied] = useState(false);
  const [registry, setRegistry] = useState<McpServer[] | null>(null);
  const [attachOpen, setAttachOpen] = useState(false);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [attachingId, setAttachingId] = useState<string | null>(null);
  // "Create new server" mode inside the attach modal (issue #7).
  const [createMode, setCreateMode] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createForm, setCreateForm] = useState<CreateServerFormState>(EMPTY_CREATE_FORM);
  const [toolsByServer, setToolsByServer] = useState<Record<string, McpToolInfo[]>>({});
  // Requested-but-unloaded servers (failed discovery) fall back to the manual
  // allowlist input instead of spinning forever.
  const [toolsRequested, setToolsRequested] = useState<Record<string, boolean>>({});

  const loadBindings = useCallback(async () => {
    setLoading(true);
    setLoadFailed(false);
    try {
      const res = await fetch(`/api/products/${productId}/mcp`, {
        credentials: "include",
        cache: "no-store",
      });
      if (res.status === 404) {
        // Registry not deployed (yet) — degrade to an empty panel.
        setBindings([]);
        return;
      }
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = await res.json();
      setBindings(Array.isArray(data) ? data : []);
    } catch {
      setLoadFailed(true);
      setBindings([]);
    } finally {
      setLoading(false);
    }
  }, [productId]);

  useEffect(() => {
    void loadBindings();
  }, [loadBindings]);

  // Probe the global registry once for admins. It feeds the attach picker,
  // per-server status_error tooltips and the tool-discovery cache. A 403
  // hides the attach button gracefully.
  useEffect(() => {
    if (!isAdmin) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/admin/mcp/servers", {
          credentials: "include",
          cache: "no-store",
        });
        if (res.status === 401 || res.status === 403) {
          if (!cancelled) setRegistryDenied(true);
          return;
        }
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setRegistry(Array.isArray(data) ? data : []);
      } catch {
        /* offline — the attach picker retries when opened */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isAdmin]);

  // Lazily discover tools for every bound server (admins only; uses the
  // discovery cache so no reconnection happens).
  useEffect(() => {
    if (!isAdmin) return;
    for (const b of bindings) {
      const serverId = b.mcp_server_id;
      if (toolsRequested[serverId]) continue;
      setToolsRequested((prev) => ({ ...prev, [serverId]: true }));
      void (async () => {
        try {
          const res = await fetch(
            `/api/admin/mcp/servers/${encodeURIComponent(serverId)}/tools`,
            { credentials: "include", cache: "no-store" },
          );
          if (!res.ok) return;
          const data = await res.json();
          if (Array.isArray(data)) {
            setToolsByServer((prev) => ({ ...prev, [serverId]: data }));
          }
        } catch {
          /* tools stay unavailable → manual allowlist fallback */
        }
      })();
    }
  }, [isAdmin, bindings, toolsRequested]);

  const registryById = new Map((registry ?? []).map((s) => [s.id, s]));

  const statusLabel = (b: McpServerBinding) => {
    const err = registryById.get(b.mcp_server_id)?.status_error;
    const base =
      b.status === "ok"
        ? (tm.statusOk ?? "ok")
        : b.status === "error"
          ? (tm.statusError ?? "error")
          : (tm.statusUnknown ?? "unknown");
    return err ? `${base} — ${err}` : base;
  };

  const patchBinding = (updated: McpServerBinding) =>
    setBindings((prev) => prev.map((x) => (x.id === updated.id ? updated : x)));

  const toggleEnabled = async (b: McpServerBinding) => {
    const next = !b.enabled;
    setBusyId(b.id);
    // Optimistic flip; reverted on failure.
    setBindings((prev) => prev.map((x) => (x.id === b.id ? { ...x, enabled: next } : x)));
    try {
      const res = await fetch(`/api/products/${productId}/mcp/${b.id}`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error((e as any)?.detail || `status ${res.status}`);
      }
      const updated = (await res.json().catch(() => null)) as McpServerBinding | null;
      if (updated) patchBinding(updated);
    } catch (e) {
      setBindings((prev) => prev.map((x) => (x.id === b.id ? { ...x, enabled: b.enabled } : x)));
      notify({
        tone: "error",
        title: tm.updateFailedTitle ?? "Update failed",
        message: e instanceof Error ? e.message : (tm.updateFailed ?? "Update failed"),
      });
    } finally {
      setBusyId(null);
    }
  };

  const saveAllowlist = async (b: McpServerBinding, allowed: string[] | null) => {
    setSavingId(b.id);
    try {
      const res = await fetch(`/api/products/${productId}/mcp/${b.id}`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allowed_tools: allowed }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error((e as any)?.detail || `status ${res.status}`);
      }
      const updated = (await res.json().catch(() => null)) as McpServerBinding | null;
      if (updated) patchBinding(updated);
      notify({ tone: "success", title: tm.allowlistSavedTitle ?? "Allowed tools saved" });
    } catch (e) {
      notify({
        tone: "error",
        title: tm.updateFailedTitle ?? "Update failed",
        message: e instanceof Error ? e.message : (tm.updateFailed ?? "Update failed"),
      });
    } finally {
      setSavingId(null);
    }
  };

  const detach = async (b: McpServerBinding) => {
    if (!confirm(tm.detachConfirm ?? "Detach this MCP server from the product?")) return;
    setBusyId(b.id);
    try {
      const res = await fetch(`/api/products/${productId}/mcp/${b.id}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error((e as any)?.detail || `status ${res.status}`);
      }
      setBindings((prev) => prev.filter((x) => x.id !== b.id));
      notify({ tone: "success", title: tm.detachedToastTitle ?? "Server detached" });
    } catch (e) {
      notify({
        tone: "error",
        title: tm.detachFailedTitle ?? "Detach failed",
        message: e instanceof Error ? e.message : (tm.detachFailed ?? "Detach failed"),
      });
    } finally {
      setBusyId(null);
    }
  };

  const openAttach = async () => {
    setAttachOpen(true);
    setRegistryLoading(true);
    try {
      const res = await fetch("/api/admin/mcp/servers", {
        credentials: "include",
        cache: "no-store",
      });
      if (res.status === 401 || res.status === 403) {
        // Not an admin after all — hide the attach affordance gracefully.
        setRegistryDenied(true);
        setAttachOpen(false);
        return;
      }
      if (!res.ok) throw new Error(`status ${res.status}`);
      setRegistry(await res.json());
    } catch {
      setRegistry(null);
      notify({
        tone: "error",
        title: tm.registryFailedTitle ?? "Registry unavailable",
        message: tm.registryFailed ?? "Failed to load the MCP server registry.",
      });
    } finally {
      setRegistryLoading(false);
    }
  };

  const attach = async (server: McpServer) => {
    setAttachingId(server.id);
    try {
      const res = await fetch(`/api/products/${productId}/mcp`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mcp_server_id: server.id, enabled: true }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error((e as any)?.detail || `status ${res.status}`);
      }
      const binding = (await res.json()) as McpServerBinding;
      setBindings((prev) => [...prev, binding]);
      setAttachOpen(false);
      notify({
        tone: "success",
        title: tm.attachedToastTitle ?? "Server attached",
        message: fmt(tm.attachedToast ?? "“{name}” tools are now available to the expert agent.", { name: server.name }),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: tm.attachFailedTitle ?? "Attach failed",
        message: e instanceof Error ? e.message : (tm.attachFailed ?? "Attach failed"),
      });
    } finally {
      setAttachingId(null);
    }
  };

  // Create a server in the global registry and bind it to this product in one
  // shot (issue #7 — no more dead end when every registry server is bound).
  const createAndAttach = async () => {
    const isHttp = createForm.transport !== "stdio";
    if (!createForm.name.trim() || (isHttp ? !createForm.url.trim() : !createForm.command.trim()))
      return;
    let headers: Record<string, string> | undefined;
    let env: Record<string, string> | undefined;
    const blocks: Array<["headers" | "env", string]> = [
      ["headers", createForm.headersText],
      ["env", createForm.envText],
    ];
    for (const [label, text] of blocks) {
      if (!text.trim()) continue;
      const { map, invalid } = parseKeyValueLines(text);
      if (invalid.length) {
        notify({
          tone: "error",
          title: tm.createFailedTitle ?? "Create failed",
          message: fmt(tm.invalidKeyValue ?? "Invalid {label} line: {line}", {
            label,
            line: invalid[0],
          }),
        });
        return;
      }
      if (label === "headers") headers = map;
      else env = map;
    }
    setCreating(true);
    try {
      const body: Record<string, unknown> = {
        name: createForm.name.trim(),
        transport: createForm.transport,
        enabled: createForm.enabled,
      };
      if (isHttp) {
        body.url = createForm.url.trim();
        if (headers) body.headers = headers;
      } else {
        body.command = createForm.command.trim();
        body.args = createForm.argsText
          .split(",")
          .map((a) => a.trim())
          .filter(Boolean);
        if (env) body.env = env;
      }
      const res = await fetch("/api/admin/mcp/servers", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error((e as any)?.detail || `status ${res.status}`);
      }
      const server = (await res.json()) as McpServer;
      setRegistry((prev) => [...(prev ?? []), server]);
      const attachRes = await fetch(`/api/products/${productId}/mcp`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mcp_server_id: server.id, enabled: true }),
      });
      if (!attachRes.ok) {
        const e = await attachRes.json().catch(() => ({}));
        throw new Error((e as any)?.detail || `status ${attachRes.status}`);
      }
      const binding = (await attachRes.json()) as McpServerBinding;
      setBindings((prev) => [...prev, binding]);
      setAttachOpen(false);
      setCreateMode(false);
      notify({
        tone: "success",
        title: tm.createdAttachedTitle ?? "Server created & attached",
        message: fmt(
          tm.createdAttached ?? "“{name}” is now available to the expert agent.",
          { name: server.name },
        ),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: tm.createFailedTitle ?? "Create failed",
        message: e instanceof Error ? e.message : (tm.createFailed ?? "Create failed"),
      });
    } finally {
      setCreating(false);
    }
  };

  const canAttach = isAdmin && !registryDenied;
  const boundServerIds = new Set(bindings.map((b) => b.mcp_server_id));
  const candidates = (registry ?? []).filter((s) => !boundServerIds.has(s.id));

  const counterText = (b: McpServerBinding) => {
    const tools = toolsByServer[b.mcp_server_id];
    if (tools) {
      return b.allowed_tools === null
        ? fmt(tm.toolsCount ?? "{n} tools", { n: tools.length })
        : fmt(tm.toolsCountOf ?? "{n} of {total} tools", {
            n: b.allowed_tools.length,
            total: tools.length,
          });
    }
    return b.allowed_tools === null
      ? (tm.allTools ?? "all tools")
      : fmt(tm.toolsCount ?? "{n} tools", { n: b.allowed_tools.length });
  };

  return (
    <section>
      <SectionHeader
        title={tm.title ?? "MCP Servers"}
        subtitle={tm.subtitle ?? ""}
        action={
          canAttach ? (
            <Button size="sm" onClick={openAttach} disabled={registryLoading}>
              <Plus size={14} weight="bold" />
              {tm.attach ?? "Attach server"}
            </Button>
          ) : undefined
        }
      />

      <div className="mt-6">
        {loading ? (
          <div className="flex items-center gap-2 text-sm text-muted">
            <Spinner /> {tm.loading ?? "Loading MCP servers…"}
          </div>
        ) : loadFailed ? (
          <div className="rounded-lg border border-dashed border-divider bg-surface px-4 py-6 text-sm text-muted">
            {tm.loadFailed ?? "Failed to load MCP bindings."}
          </div>
        ) : bindings.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-divider bg-surface px-6 py-10 text-center">
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-surface-2 text-muted">
              <Plugs size={20} weight="regular" />
            </div>
            <p className="text-sm text-muted">{tm.empty ?? "No MCP servers bound to this product."}</p>
            {canAttach && (
              <div className="mt-4">
                <Button size="sm" variant="subtle" onClick={openAttach}>
                  <Plus size={14} weight="bold" />
                  {tm.attach ?? "Attach server"}
                </Button>
              </div>
            )}
          </div>
        ) : (
          <Card className="overflow-hidden">
            <ul>
              {bindings.map((b) => {
                const isExpanded = expandedId === b.id;
                const tools = toolsByServer[b.mcp_server_id] ?? null;
                return (
                  <li key={b.id} className="border-t border-divider first:border-t-0">
                    <div className="flex items-center gap-3 px-4 py-3">
                      <McpStatusDot status={b.status} title={statusLabel(b)} />
                      <button
                        type="button"
                        onClick={() => setExpandedId(isExpanded ? null : b.id)}
                        className="min-w-0 flex-1 text-left"
                        aria-expanded={isExpanded}
                      >
                        <span className="flex flex-wrap items-center gap-2">
                          <span className="truncate text-sm font-medium text-ink">{b.name}</span>
                          <Tag tone={transportTone(b.transport)}>{b.transport}</Tag>
                          {!b.enabled && <Tag tone="yellow">{tm.disabled ?? "off"}</Tag>}
                        </span>
                        <span className="mt-0.5 block text-xs text-muted">{counterText(b)}</span>
                      </button>
                      <IconButton
                        aria-label={tm.allowlist ?? "Allowed tools"}
                        title={tm.allowlist ?? "Allowed tools"}
                        className="h-8 w-8"
                        onClick={() => setExpandedId(isExpanded ? null : b.id)}
                      >
                        {isExpanded ? <CaretUp size={14} weight="bold" /> : <CaretDown size={14} weight="bold" />}
                      </IconButton>
                      <Switch
                        checked={b.enabled}
                        onChange={() => toggleEnabled(b)}
                        disabled={busyId === b.id}
                        label={tm.toggleLabel ?? "Enable or disable this MCP server"}
                      />
                      <IconButton
                        aria-label={tm.detach ?? "Detach"}
                        title={tm.detach ?? "Detach"}
                        disabled={busyId === b.id}
                        onClick={() => detach(b)}
                      >
                        {busyId === b.id ? <Spinner /> : <Trash size={14} weight="regular" />}
                      </IconButton>
                    </div>
                    {isExpanded && (
                      <div className="border-t border-divider bg-surface-2/60 px-4 pt-3">
                        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">
                          {tm.allowlist ?? "Allowed tools"}
                        </p>
                        {tools === null && isAdmin && !toolsRequested[b.mcp_server_id] ? (
                          <p className="pb-4 text-xs text-muted">
                            <Spinner className="mr-1 inline align-[-2px]" />
                            {tm.toolsLoading ?? "Discovering tools…"}
                          </p>
                        ) : (
                          <AllowlistEditor
                            binding={b}
                            tools={tools}
                            saving={savingId === b.id}
                            onSave={(allowed) => saveAllowlist(b, allowed)}
                            tm={tm}
                            fmt={fmt}
                          />
                        )}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </Card>
        )}
      </div>

      {/* Attach picker — global registry, with an inline "create new server" mode */}
      <Modal
        open={attachOpen}
        onClose={() => {
          setAttachOpen(false);
          setCreateMode(false);
        }}
        title={
          createMode
            ? (tm.createTitle ?? "New MCP server")
            : (tm.attachTitle ?? "Attach MCP server")
        }
        footer={null}
      >
        {createMode ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void createAndAttach();
            }}
            className="grid gap-4"
          >
            <div className="grid gap-4 md:grid-cols-[1fr_180px]">
              <div>
                <Label>{tm.nameLabel ?? "Name"}</Label>
                <Input
                  value={createForm.name}
                  onChange={(e) =>
                    setCreateForm((p) => ({ ...p, name: e.target.value }))
                  }
                  placeholder={tm.namePlaceholder ?? "e.g. Context7 docs"}
                  maxLength={128}
                  required
                  autoFocus
                />
              </div>
              <div>
                <Label>{tm.transport ?? "Transport"}</Label>
                <Select
                  value={createForm.transport}
                  onChange={(e) =>
                    setCreateForm((p) => ({
                      ...p,
                      transport: e.target.value as McpTransport,
                    }))
                  }
                >
                  <option value="http">{tm.transportHttp ?? "HTTP"}</option>
                  <option value="sse">{tm.transportSse ?? "SSE"}</option>
                  <option value="stdio">{tm.transportStdio ?? "Stdio"}</option>
                </Select>
              </div>
            </div>

            {createForm.transport !== "stdio" ? (
              <>
                <div>
                  <Label>{tm.url ?? "URL"}</Label>
                  <Input
                    type="url"
                    value={createForm.url}
                    onChange={(e) =>
                      setCreateForm((p) => ({ ...p, url: e.target.value }))
                    }
                    placeholder="https://mcp.example.com/mcp"
                    pattern="https?://.+"
                    title="http:// or https:// only"
                    required
                  />
                </div>
                <div>
                  <Label>{tm.headers ?? "Headers (key=value per line)"}</Label>
                  <Textarea
                    value={createForm.headersText}
                    onChange={(e) =>
                      setCreateForm((p) => ({ ...p, headersText: e.target.value }))
                    }
                    placeholder="Authorization=Bearer …"
                    rows={3}
                    spellCheck={false}
                  />
                </div>
              </>
            ) : (
              <>
                <div className="grid gap-4 md:grid-cols-[200px_1fr]">
                  <div>
                    <Label>{tm.command ?? "Command"}</Label>
                    <Input
                      value={createForm.command}
                      onChange={(e) =>
                        setCreateForm((p) => ({ ...p, command: e.target.value }))
                      }
                      placeholder="npx"
                      required
                    />
                  </div>
                  <div>
                    <Label>{tm.args ?? "Arguments (comma-separated)"}</Label>
                    <Input
                      value={createForm.argsText}
                      onChange={(e) =>
                        setCreateForm((p) => ({ ...p, argsText: e.target.value }))
                      }
                      placeholder="-y, @modelcontextprotocol/server-everything"
                      className="font-mono text-sm"
                    />
                  </div>
                </div>
                <div>
                  <Label>{tm.env ?? "Environment (key=value per line)"}</Label>
                  <Textarea
                    value={createForm.envText}
                    onChange={(e) =>
                      setCreateForm((p) => ({ ...p, envText: e.target.value }))
                    }
                    placeholder="API_KEY=…"
                    rows={3}
                    spellCheck={false}
                  />
                </div>
              </>
            )}

            <div className="flex items-center gap-3 rounded-md border border-divider bg-surface-2 p-3">
              <Switch
                checked={createForm.enabled}
                onChange={(next) =>
                  setCreateForm((p) => ({ ...p, enabled: next }))
                }
                label={tm.enabledLabel ?? "Enabled"}
              />
              <span className="text-sm text-ink">{tm.enabledLabel ?? "Enabled"}</span>
            </div>

            <div className="flex items-center justify-end gap-2">
              <Button type="button" variant="ghost" onClick={() => setCreateMode(false)}>
                {tm.cancel ?? "Cancel"}
              </Button>
              <Button
                type="submit"
                disabled={
                  creating ||
                  !createForm.name.trim() ||
                  (createForm.transport !== "stdio"
                    ? !createForm.url.trim()
                    : !createForm.command.trim())
                }
              >
                {creating ? <Spinner /> : <Plus size={14} weight="bold" />}
                {tm.createAndAttach ?? "Create & attach"}
              </Button>
            </div>
          </form>
        ) : registryLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted">
            <Spinner /> {tm.registryLoading ?? "Loading registry…"}
          </div>
        ) : registry === null ? (
          <p className="text-sm text-muted">{tm.registryFailed ?? "Failed to load the MCP server registry."}</p>
        ) : candidates.length === 0 ? (
          <div className="text-sm text-muted">
            <p>{tm.noCandidates ?? "All registry servers are already bound."}</p>
            <div className="mt-3">
              <Button size="sm" onClick={() => setCreateMode(true)}>
                <Plus size={14} weight="bold" />
                {tm.createServer ?? "Create new server"}
              </Button>
            </div>
          </div>
        ) : (
          <>
            <ul className="flex flex-col gap-2">
              {candidates.map((s) => (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => attach(s)}
                    disabled={attachingId === s.id}
                    className="flex w-full items-center gap-3 rounded-md border border-divider bg-surface px-3 py-2.5 text-left transition-colors hover:bg-surface-2 disabled:opacity-60"
                  >
                    {attachingId === s.id ? <Spinner /> : <McpStatusDot status={s.status} />}
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="truncate text-sm font-medium text-ink">{s.name}</span>
                        <Tag tone={transportTone(s.transport)}>{s.transport}</Tag>
                        {!s.enabled && <Tag tone="yellow">{tm.disabled ?? "off"}</Tag>}
                      </span>
                      {(s.url || s.command) && (
                        <span className="mt-0.5 block truncate font-mono text-xs text-muted">
                          {s.transport !== "stdio"
                            ? s.url
                            : [s.command, ...(s.args ?? [])].join(" ")}
                        </span>
                      )}
                    </span>
                    <Plus size={14} weight="bold" className="shrink-0 text-muted" />
                  </button>
                </li>
              ))}
            </ul>
            <div className="mt-4 border-t border-divider pt-4">
              <Button size="sm" variant="subtle" onClick={() => setCreateMode(true)}>
                <Plus size={14} weight="bold" />
                {tm.createServer ?? "Create new server"}
              </Button>
            </div>
          </>
        )}
      </Modal>
    </section>
  );
}
