"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  ArrowsCounterClockwise,
  Brain,
  Copy,
  FileText,
  Gear,
  GitBranch,
  Globe,
  Key,
  PencilSimple,
  Plug,
  Plugs,
  Plus,
  Rocket,
  SealCheck,
  Shield,
  Spinner,
  Trash,
  UserCircleGear,
  Wrench,
} from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { AuthGuard } from "@/components/AuthGuard";
import { useNotifications } from "@/contexts/NotificationContext";
import {
  Banner,
  Button,
  Card,
  Input,
  Label,
  Modal,
  SectionHeader,
  Select,
  Spinner as SpinnerIcon,
  Switch,
  Tag,
  Textarea,
  cn,
} from "@/components/ui";
import { McpStatusDot } from "@/components/mcp/McpStatusDot";
import type {
  ApiToken,
  McpServer,
  McpTestResult,
  McpTransport,
  User,
  UserCreateResult,
  UserRole,
} from "@/lib/types";
import { useLanguage } from "@/contexts/LanguageContext";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

type Section =
  | "models"
  | "ssl"
  | "git"
  | "confluence"
  | "integrations"
  | "mcp"
  | "prompts"
  | "memory"
  | "timeouts"
  | "users"
  | "tokens";

const SECTIONS: { key: Section; icon: typeof Gear }[] = [
  { key: "models", icon: Rocket },
  { key: "ssl", icon: Shield },
  { key: "git", icon: GitBranch },
  { key: "confluence", icon: Globe },
  { key: "integrations", icon: Plug },
  { key: "mcp", icon: Plugs },
  { key: "prompts", icon: FileText },
  { key: "memory", icon: Brain },
  { key: "timeouts", icon: Wrench },
  { key: "users", icon: UserCircleGear },
  { key: "tokens", icon: Key },
];

/**
 * Shared admin API helpers. Each request sends the session cookie; on a 401
 * (session expired mid-panel) the user is redirected to /login silently.
 * Errors are thrown so callers can route them to a toast via `notify`.
 */
function useAdminApi() {
  const { notify } = useNotifications();
  const router = useRouter();

  const on401 = useCallback(() => {
    router.replace("/login?next=/admin");
  }, [router]);

  const getJson = useCallback(
    async <T,>(url: string): Promise<T> => {
      const res = await fetch(url, { credentials: "include", cache: "no-store" });
      if (res.status === 401) {
        on401();
        throw new Error("Session expired");
      }
      if (!res.ok) throw new Error(`GET ${url} failed (${res.status})`);
      return (await res.json()) as T;
    },
    [on401],
  );

  const putJson = useCallback(
    async (url: string, body: unknown): Promise<unknown> => {
      const res = await fetch(url, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status === 401) {
        on401();
        throw new Error("Session expired");
      }
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e?.detail || `Save failed (${res.status})`);
      }
      return res.json().catch(() => ({}));
    },
    [on401],
  );

  const postJson = useCallback(
    async (url: string, body?: unknown): Promise<unknown> => {
      const res = await fetch(url, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      if (res.status === 401) {
        on401();
        throw new Error("Session expired");
      }
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e?.detail || `Request failed (${res.status})`);
      }
      return res.json().catch(() => ({}));
    },
    [on401],
  );

  const del = useCallback(
    async (url: string): Promise<void> => {
      const res = await fetch(url, { method: "DELETE", credentials: "include" });
      if (res.status === 401) {
        on401();
        throw new Error("Session expired");
      }
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e?.detail || `Delete failed (${res.status})`);
      }
    },
    [on401],
  );

  return { getJson, putJson, postJson, del, notify };
}

/* ------------------------------------------------------------------ */
/* Field row                                                           */
/* ------------------------------------------------------------------ */

function Field({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  secret,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  secret?: boolean;
}) {
  return (
    <div>
      <Label>{label}</Label>
      <Input
        type={secret ? "password" : type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Models section                                                      */
/* ------------------------------------------------------------------ */

type ModelTask = "docgen" | "expert" | "summary" | "embedder";

interface ModelCfg {
  model: string;
  base_url: string;
  api_key: string;
  // UI-only: true when a key is already stored (redacted on the server).
  // Never sent on save.
  hasApiKey: boolean;
  // Optional per-model prompt-token budget (models.<task>.max_prompt_tokens).
  // Empty string = use the default (no override).
  max_prompt_tokens: string;
  dimensions?: string;
}
type ModelsConfig = Record<ModelTask, ModelCfg>;

// Raw shape returned by GET /api/admin/models (and other setting groups):
// { group, settings, resolved }. The Models section edits the `resolved` map.
interface ModelGroupResponse {
  group: string;
  settings: Record<string, unknown>;
  resolved: Partial<Record<ModelTask, ModelResolvedEntry>>;
}
interface ModelResolvedEntry {
  model: string | null;
  base_url: string | null;
  api_key: string | null;
  hasApiKey: boolean;
  max_prompt_tokens: number | null;
  dimensions: number | null;
}

const DEFAULT_MODEL_CFG: ModelCfg = {
  model: "",
  base_url: "",
  api_key: "",
  hasApiKey: false,
  max_prompt_tokens: "",
  dimensions: "",
};

const MODEL_TASKS: ModelTask[] = [
  "docgen",
  "expert",
  "summary",
  "embedder",
];

function ModelsSection() {
  const { getJson, putJson, postJson, notify } = useAdminApi();
  const { messages, fmt } = useLanguage();
  const t = messages?.admin ?? {};
  const tm = t?.models ?? {};
  const [cfg, setCfg] = useState<ModelsConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [testing, setTesting] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      // The backend returns { group, settings, resolved }; the editable
      // per-task config lives under `resolved` (secrets redacted as
      // hasApiKey). Map it into the editable ModelsConfig shape.
      const data = await getJson<ModelGroupResponse>("/api/admin/models");
      const resolved = data?.resolved ?? {};
      const next = {} as ModelsConfig;
      for (const task of MODEL_TASKS) {
        const r = resolved[task];
        next[task] = {
          model: r?.model ?? "",
          base_url: r?.base_url ?? "",
          api_key: "",
          hasApiKey: Boolean(r?.hasApiKey),
          max_prompt_tokens:
            r?.max_prompt_tokens != null ? String(r.max_prompt_tokens) : "",
          dimensions: r?.dimensions != null ? String(r.dimensions) : "",
        };
      }
      setCfg(next);
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
  }, [load]);

  const update = (task: ModelTask, key: keyof ModelCfg, v: string) => {
    setCfg((prev) => {
      const cur = prev?.[task] ?? DEFAULT_MODEL_CFG;
      return prev ? { ...prev, [task]: { ...cur, [key]: v } } : prev;
    });
  };

  const save = async (task: ModelTask) => {
    if (!cfg) return;
    setSaving(task);
    try {
      const cur = cfg[task] ?? DEFAULT_MODEL_CFG;
      // The PUT contract expects flat setting keys `models.<task>.<field>`.
      // Only send api_key when the admin typed a new value, to avoid
      // overwriting a stored secret with an empty string. The `provider` field
      // is no longer surfaced or sent; the backend always treats every model
      // as OpenAI-compatible.
      const body: Record<string, string> = {
        [`models.${task}.model`]: cur.model,
        [`models.${task}.base_url`]: cur.base_url,
        [`models.${task}.max_prompt_tokens`]: cur.max_prompt_tokens,
        [`models.${task}.dimensions`]: cur.dimensions ?? "",
      };
      if (cur.api_key) body[`models.${task}.api_key`] = cur.api_key;
      await putJson("/api/admin/models", body);
      await load();
      notify({ tone: "success", title: fmt(tm.savedToast ?? "Saved {task} model config.", { task }) });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSaving(null);
    }
  };

  const test = async (task: ModelTask) => {
    setTesting(task);
    try {
      // Backend returns { success, message } (not { ok }). Accept `ok` too for safety.
      const res = (await postJson("/api/admin/models/test", { task })) as {
        success?: boolean;
        ok?: boolean;
        message?: string;
      };
      const ok = Boolean(res.success ?? res.ok);
      notify({
        tone: ok ? "success" : "error",
        title: ok ? (tm.connOkTitle ?? "Connection OK") : (t.testFailedTitle ?? "Test failed"),
        message: res.message || (ok ? (tm.connOkTitle ?? "Connection OK") : (tm.testFailedToast ?? "Test failed")),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.testFailedTitle ?? "Test failed",
        message: e instanceof Error ? e.message : (t.testFailed ?? "Test failed"),
      });
    } finally {
      setTesting(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tm.loading ?? "Loading models…"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {tm.intro ?? ""}
      </p>
      {MODEL_TASKS.map((task) => {
        const c = cfg?.[task] ?? DEFAULT_MODEL_CFG;
        return (
          <Card key={task} className="p-5">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-editorial text-base tracking-tight text-ink">
                {task}
              </h3>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <Field
                label={tm.model ?? "Model"}
                value={c.model}
                onChange={(v) => update(task, "model", v)}
                placeholder="e.g. qwen/qwen3.6-27b"
              />
              <Field
                label={tm.baseUrl ?? "Base URL"}
                value={c.base_url}
                onChange={(v) => update(task, "base_url", v)}
                placeholder="http://localhost:1234/v1"
              />
              <Field
                label={tm.apiKey ?? "API key"}
                value={c.api_key}
                onChange={(v) => update(task, "api_key", v)}
                placeholder={
                  c.hasApiKey
                    ? (tm.keyStored ?? "stored · type to replace")
                    : (tm.keyPlaceholder ?? "type to set")
                }
                secret
              />
              {task === "embedder" && (
                <div className="md:col-span-2">
                  <Field
                    label="Vector dimensions (e.g. 1536, 1024, 768, 384)"
                    value={c.dimensions ?? ""}
                    onChange={(v) =>
                      update(
                        task,
                        "dimensions",
                        v.replace(/[^0-9]/g, "").slice(0, 5),
                      )
                    }
                    placeholder="Auto-detected if left empty (default: 768)"
                  />
                  <p className="mt-1 text-xs text-muted">
                    Optional. Specify custom embedding vector dimensions if using a non-768 model (e.g. 1536 for text-embedding-3-small or 1024 for qwen-embedding).
                  </p>
                </div>
              )}
              <div className="md:col-span-2">
                <Field
                  label={tm.maxPromptTokens ?? "Max prompt tokens"}
                  value={c.max_prompt_tokens}
                  onChange={(v) =>
                    update(
                      task,
                      "max_prompt_tokens",
                      // Allow only non-negative integers; ignore the rest.
                      v.replace(/[^0-9]/g, "").slice(0, 9),
                    )
                  }
                  placeholder={
                    tm.maxPromptTokensPlaceholder ??
                      "leave empty for default (200000)"
                  }
                />
                <p className="mt-1 text-xs text-muted">
                  {tm.maxPromptTokensHint ??
                    "Optional. Prompt-token limit for this model. Leave empty to use the default."}
                </p>
              </div>
            </div>
            <div className="mt-4 flex items-center gap-2">
              <Button
                size="sm"
                onClick={() => save(task)}
                disabled={saving === task}
              >
                {saving === task ? (
                  <SpinnerIcon />
                ) : (
                  <Gear size={14} weight="regular" />
                )}
                {tm.save ?? "Save"}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => test(task)}
                disabled={testing === task}
              >
                {testing === task ? (
                  <SpinnerIcon />
                ) : (
                  <Wrench size={14} weight="regular" />
                )}
                {tm.test ?? "Test"}
              </Button>
            </div>
          </Card>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Git credentials section                                             */
/* ------------------------------------------------------------------ */

const GIT_HOSTS = ["github", "gitlab"] as const;
type GitHost = (typeof GIT_HOSTS)[number];
interface GitAccount {
  url: string;
  token: string;
  // UI-only: true when a token is already stored (redacted on the server).
  hasToken: boolean;
}
type GitConfig = Record<GitHost, GitAccount[]>;

// GET /api/admin/git returns { group, settings, resolved }; the editable
// per-host accounts live under `resolved` (a list per host; token redacted
// as hasToken).
interface GitGroupResponse {
  group: string;
  settings: Record<string, unknown>;
  resolved: Partial<
    Record<GitHost, Array<{ url: string | null; token: string | null; hasToken: boolean }>>
  >;
}

const DEFAULT_GIT_ACCOUNT: GitAccount = { url: "", token: "", hasToken: false };

function GitSection() {
  const { getJson, putJson, postJson, notify } = useAdminApi();
  const { messages, fmt } = useLanguage();
  const t = messages?.admin ?? {};
  const tg = t?.git ?? {};
  const [cfg, setCfg] = useState<GitConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [testing, setTesting] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<GitGroupResponse>("/api/admin/git");
      const resolved = data?.resolved ?? {};
      const next = {} as GitConfig;
      for (const host of GIT_HOSTS) {
        const rows = Array.isArray(resolved[host]) ? (resolved[host] as GitAccount[]) : [];
        next[host] = rows.map((r) => ({
          url: r?.url ?? "",
          token: "",
          hasToken: Boolean(r?.hasToken),
        }));
      }
      setCfg(next);
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const update = (host: GitHost, idx: number, key: keyof GitAccount, v: string) =>
    setCfg((prev) => {
      if (!prev) return prev;
      const rows = [...(prev[host] ?? [])];
      rows[idx] = { ...(rows[idx] ?? DEFAULT_GIT_ACCOUNT), [key]: v };
      return { ...prev, [host]: rows };
    });

  const addAccount = (host: GitHost) =>
    setCfg((prev) => {
      if (!prev) return prev;
      const rows = [...(prev[host] ?? [])];
      rows.push({ ...DEFAULT_GIT_ACCOUNT });
      return { ...prev, [host]: rows };
    });

  const removeAccount = (host: GitHost, idx: number) =>
    setCfg((prev) => {
      if (!prev) return prev;
      const rows = [...(prev[host] ?? [])];
      rows.splice(idx, 1);
      return { ...prev, [host]: rows };
    });

  const save = async (host: GitHost) => {
    if (!cfg) return;
    setSaving(host);
    try {
      const accounts = (cfg[host] ?? []).map((a) => ({
        url: a.url,
        // Empty token means "keep the existing stored token for this URL".
        token: a.token,
      }));
      await putJson("/api/admin/git", { accounts: { [host]: accounts } });
      await load();
      notify({ tone: "success", title: fmt(tg.savedToast ?? "Saved {host} credentials.", { host }) });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSaving(null);
    }
  };

  const test = async (host: GitHost) => {
    setTesting(host);
    try {
      const res = (await postJson("/api/admin/git/test", { host })) as {
        success?: boolean;
        ok?: boolean;
        message?: string;
      };
      const ok = Boolean(res.success ?? res.ok);
      notify({
        tone: ok ? "success" : "error",
        title: ok ? (tg.reachableTitle ?? "Reachable") : (t.testFailedTitle ?? "Test failed"),
        message: res.message || (ok ? (tg.reachable ?? "Reachable") : (t.testFailed ?? "Test failed")),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.testFailedTitle ?? "Test failed",
        message: e instanceof Error ? e.message : (t.testFailed ?? "Test failed"),
      });
    } finally {
      setTesting(null);
    }
  };

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tg.loading ?? "Loading…"}
      </div>
    );

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {tg.intro ?? ""}
      </p>
      {GIT_HOSTS.map((host) => {
        const accounts = cfg?.[host] ?? [];
        return (
          <Card key={host} className="p-5">
            <h3 className="mb-1 font-editorial text-base tracking-tight text-ink capitalize">
              {host}
            </h3>
            <p className="mb-4 text-[13px] text-muted">
              {tg.accountsHint ?? ""}
            </p>
            <div className="space-y-3">
              {accounts.map((a, idx) => (
                <div key={idx} className="grid gap-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
                  <Field
                    label={tg.accountUrl ?? "URL"}
                    value={a.url}
                    onChange={(v) => update(host, idx, "url", v)}
                    placeholder={tg.accountUrlPlaceholder ?? "empty = public cloud"}
                  />
                  <Field
                    label={tg.accessToken ?? "Access token"}
                    value={a.token}
                    onChange={(v) => update(host, idx, "token", v)}
                    placeholder={
                      a.hasToken
                        ? (tg.tokenStored ?? "stored · type to replace")
                        : (tg.tokenPlaceholder ?? "type to set")
                    }
                    secret
                  />
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => removeAccount(host, idx)}
                    aria-label={tg.removeAccount ?? "Remove account"}
                  >
                    <Trash size={14} weight="regular" />
                  </Button>
                </div>
              ))}
              <Button size="sm" variant="ghost" onClick={() => addAccount(host)}>
                <Plus size={14} weight="regular" /> {tg.addAccount ?? "Add account"}
              </Button>
            </div>
            <div className="mt-4 flex items-center gap-2">
              <Button size="sm" onClick={() => save(host)} disabled={saving === host}>
                {saving === host ? (
                  <SpinnerIcon />
                ) : (
                  <Gear size={14} weight="regular" />
                )}{" "}
                {tg.save ?? "Save"}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => test(host)} disabled={testing === host}>
                {testing === host ? (
                  <SpinnerIcon />
                ) : (
                  <Wrench size={14} weight="regular" />
                )}{" "}
                {tg.test ?? "Test"}
              </Button>
            </div>
          </Card>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Confluence section                                                  */
/* ------------------------------------------------------------------ */

interface ConfluenceCfg {
  base_url: string;
  token: string;
  space: string;
  // UI-only: true when a token is already stored (redacted on the server).
  hasToken: boolean;
}

// GET /api/admin/confluence returns { group, settings, resolved }; the
// editable creds live under `resolved` (token redacted as hasToken).
interface ConfluenceGroupResponse {
  group: string;
  settings: Record<string, unknown>;
  resolved: {
    base_url: string | null;
    token: string | null;
    space: string | null;
    hasToken: boolean;
  } | null;
}

function ConfluenceSection() {
  const { getJson, putJson, postJson, notify } = useAdminApi();
  const { messages } = useLanguage();
  const t = messages?.admin ?? {};
  const tc = t?.confluence ?? {};
  const [cfg, setCfg] = useState<ConfluenceCfg>({
    base_url: "",
    token: "",
    space: "",
    hasToken: false,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<ConfluenceGroupResponse>("/api/admin/confluence");
      const r = data?.resolved;
      setCfg({
        base_url: r?.base_url ?? "",
        token: "",
        space: r?.space ?? "",
        hasToken: Boolean(r?.hasToken),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      // PUT contract expects flat keys `confluence.{base_url,token,space}`.
      // Only send the token when the admin typed a new value.
      const body: Record<string, string> = {
        "confluence.base_url": cfg.base_url,
        "confluence.space": cfg.space,
      };
      if (cfg.token) body["confluence.token"] = cfg.token;
      await putJson("/api/admin/confluence", body);
      await load();
      notify({ tone: "success", title: tc.savedToast ?? "Saved Confluence config." });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    setTesting(true);
    try {
      const res = (await postJson("/api/admin/confluence/test", {})) as {
        success?: boolean;
        ok?: boolean;
        message?: string;
      };
      const ok = Boolean(res.success ?? res.ok);
      notify({
        tone: ok ? "success" : "error",
        title: ok ? (tc.reachableTitle ?? "Reachable") : (t.testFailedTitle ?? "Test failed"),
        message: res.message || (ok ? (tc.reachable ?? "Reachable") : (t.testFailed ?? "Test failed")),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.testFailedTitle ?? "Test failed",
        message: e instanceof Error ? e.message : (t.testFailed ?? "Test failed"),
      });
    } finally {
      setTesting(false);
    }
  };

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tc.loading ?? "Loading…"}
      </div>
    );

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <div className="grid gap-4 md:grid-cols-3">
          <Field
            label={tc.baseUrl ?? "Base URL"}
            value={cfg.base_url}
            onChange={(v) => setCfg((p) => ({ ...p, base_url: v }))}
            placeholder="https://company.atlassian.net"
          />
          <Field
            label={tc.apiToken ?? "API token"}
            value={cfg.token}
            onChange={(v) => setCfg((p) => ({ ...p, token: v }))}
            placeholder={
              cfg.hasToken
                ? (tc.tokenStored ?? "stored · type to replace")
                : (tc.tokenPlaceholder ?? "type to set")
            }
            secret
          />
          <Field
            label={tc.spaceKey ?? "Space key"}
            value={cfg.space}
            onChange={(v) => setCfg((p) => ({ ...p, space: v }))}
            placeholder="e.g. ENG"
          />
        </div>
        <div className="mt-4 flex items-center gap-2">
          <Button size="sm" onClick={save} disabled={saving}>
            {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}{" "}
            {tc.save ?? "Save"}
          </Button>
          <Button size="sm" variant="ghost" onClick={test} disabled={testing}>
            {testing ? <SpinnerIcon /> : <Wrench size={14} weight="regular" />}{" "}
            {tc.test ?? "Test"}
          </Button>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* SSL / TLS section                                                   */
/* ------------------------------------------------------------------ */

interface SslSettingView {
  value: string | null;
  encrypted: boolean;
  hasKey: boolean;
}
// GET /api/admin/ssl returns { group, settings } with raw `ssl.*` keys.
interface SslGroupResponse {
  group: string;
  settings: Record<string, SslSettingView>;
}

function SslSection() {
  const { getJson, putJson, notify } = useAdminApi();
  const { messages, fmt } = useLanguage();
  const t = messages?.admin ?? {};
  const ts = t?.ssl ?? {};
  const [caBundle, setCaBundle] = useState("");
  const [verify, setVerify] = useState(true);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<SslGroupResponse>("/api/admin/ssl");
      const s = data?.settings ?? {};
      setCaBundle(s["ssl.ca_bundle"]?.value ?? "");
      // The verify setting defaults to ON when unset. Only treat an explicit
      // falsy value ("false"/"0"/…) as OFF.
      const raw = s["ssl.verify"]?.value;
      setVerify(raw == null ? true : !/^(0|false|f|no|n|off)$/i.test(raw.trim()));
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      // PUT contract expects flat keys `ssl.ca_bundle` / `ssl.verify`.
      const body: Record<string, string> = {
        "ssl.ca_bundle": caBundle.trim(),
        "ssl.verify": verify ? "true" : "false",
      };
      await putJson("/api/admin/ssl", body);
      await load();
      notify({ tone: "success", title: ts.savedToast ?? "Saved SSL / TLS config." });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {ts.loading ?? "Loading…"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {ts.intro ?? ""}
      </p>
      <Card className="p-5">
        <div className="space-y-4">
          <div>
            <Label>{ts.caBundle ?? "CA bundle path"}</Label>
            <Input
              value={caBundle}
              onChange={(e) => setCaBundle(e.target.value)}
              placeholder={ts.caBundlePlaceholder ?? "/absolute/path/to/corporate-root.pem"}
              className="font-mono text-sm"
            />
            <p className="mt-1.5 text-xs text-muted">{ts.caBundleHint ?? ""}</p>
          </div>
          <div className="flex items-start gap-3 rounded-md border border-divider bg-surface-2 p-3">
            <Switch checked={verify} onChange={setVerify} label={ts.verify ?? "Verify TLS certificates"} />
            <div className="flex-1">
              <div className="text-sm font-medium text-ink">
                {ts.verify ?? "Verify TLS certificates"}
              </div>
              <p className="mt-0.5 text-xs text-muted">{ts.verifyHint ?? ""}</p>
            </div>
            <Tag tone={verify ? "green" : "red"}>
              {fmt(ts.current ?? "Current: verification {state}", { state: verify ? "ON" : "OFF" })}
            </Tag>
          </div>
          {!verify && (
            <Banner tone="warning">{ts.warningOff ?? ""}</Banner>
          )}
        </div>
        <div className="mt-4 flex items-center gap-2">
          <Button size="sm" onClick={save} disabled={saving}>
            {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}{" "}
            {ts.save ?? "Save"}
          </Button>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Integrations section (JSON editor)                                  */
/* ------------------------------------------------------------------ */

function IntegrationsSection() {
  const { getJson, putJson, postJson, notify } = useAdminApi();
  const { messages } = useLanguage();
  const t = messages?.admin ?? {};
  const ti = t?.integrations ?? {};
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const data = await getJson<unknown>("/api/admin/integrations");
        setText(JSON.stringify(data ?? {}, null, 2));
      } catch (e) {
        notify({
          tone: "error",
          title: t.loadFailedTitle ?? "Load failed",
          message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
        });
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const parsed = JSON.parse(text || "{}");
      await putJson("/api/admin/integrations", parsed);
      notify({ tone: "success", title: ti.savedToast ?? "Saved integrations config." });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (ti.invalidJson ?? "Invalid JSON"),
      });
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    setTesting(true);
    try {
      const res = (await postJson("/api/admin/integrations/test", {})) as {
        success?: boolean;
        ok?: boolean;
        message?: string;
      };
      const ok = Boolean(res.success ?? res.ok);
      notify({
        tone: ok ? "success" : "error",
        title: ok ? (ti.okTitle ?? "OK") : (t.testFailedTitle ?? "Test failed"),
        message: res.message || (ok ? (ti.ok ?? "OK") : (t.testFailed ?? "Test failed")),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.testFailedTitle ?? "Test failed",
        message: e instanceof Error ? e.message : (t.testFailed ?? "Test failed"),
      });
    } finally {
      setTesting(false);
    }
  };

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {ti.loading ?? "Loading…"}
      </div>
    );

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {ti.intro ?? ""}
      </p>
      <Card className="p-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={16}
          spellCheck={false}
          className="w-full rounded-md border border-divider bg-surface-2 p-3 font-mono text-xs leading-relaxed text-ink focus:border-ink focus:outline-none"
        />
      </Card>
      <div className="flex items-center gap-2">
        <Button size="sm" onClick={save} disabled={saving}>
          {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />} {ti.save ?? "Save"}
        </Button>
        <Button size="sm" variant="ghost" onClick={test} disabled={testing}>
          {testing ? <SpinnerIcon /> : <Wrench size={14} weight="regular" />}{" "}
          {ti.test ?? "Test"}
        </Button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* MCP servers section (global registry, wave C)                      */
/* ------------------------------------------------------------------ */

interface McpFormState {
  name: string;
  transport: McpTransport;
  url: string;
  command: string;
  argsText: string;
  headersText: string;
  envText: string;
  enabled: boolean;
}

const EMPTY_MCP_FORM: McpFormState = {
  name: "",
  transport: "http",
  url: "",
  command: "",
  argsText: "",
  headersText: "",
  envText: "",
  enabled: true,
};

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

function McpSection() {
  const { getJson, putJson, postJson, del, notify } = useAdminApi();
  const { messages, fmt } = useLanguage();
  const t = messages?.admin ?? {};
  const tc = messages?.common ?? {};
  const tm = t?.mcp ?? {};

  const [servers, setServers] = useState<McpServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  // Create / edit form state.
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<McpServer | null>(null);
  const [form, setForm] = useState<McpFormState>(EMPTY_MCP_FORM);
  const [saving, setSaving] = useState(false);

  // Last test result (viewed in a modal: ok/detail + discovered tools).
  const [testView, setTestView] = useState<{
    server: McpServer;
    result: McpTestResult;
  } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<McpServer[] | { servers: McpServer[] }>(
        "/api/admin/mcp/servers",
      );
      setServers(Array.isArray(data) ? data : (data?.servers ?? []));
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
  }, [load]);

  const openCreate = () => {
    setEditing(null);
    setForm(EMPTY_MCP_FORM);
    setFormOpen(true);
  };

  const openEdit = (s: McpServer) => {
    setEditing(s);
    setForm({
      name: s.name,
      transport: s.transport,
      url: s.url ?? "",
      command: s.command ?? "",
      argsText: (s.args ?? []).join(", "),
      // Secrets are masked on read — leave the header/env editors empty and
      // only send values when typed ("type to replace" pattern).
      headersText: "",
      envText: "",
      enabled: s.enabled,
    });
    setFormOpen(true);
  };

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    if (saving || !form.name.trim()) return;
    // Parse the key=value blocks; abort with a toast on malformed lines so
    // garbage never reaches the backend.
    let headers: Record<string, string> | undefined;
    let env: Record<string, string> | undefined;
    const blocks: Array<["headers" | "env", string]> = [
      ["headers", form.headersText],
      ["env", form.envText],
    ];
    for (const [label, text] of blocks) {
      if (!text.trim()) continue;
      const { map, invalid } = parseKeyValueLines(text);
      if (invalid.length) {
        notify({
          tone: "error",
          title: t.saveFailedTitle ?? "Save failed",
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
    setSaving(true);
    try {
      const body: Record<string, unknown> = {
        name: form.name.trim(),
        transport: form.transport,
        enabled: form.enabled,
      };
      if (form.transport === "http") {
        body.url = form.url.trim();
        if (headers) body.headers = headers;
      } else {
        body.command = form.command.trim();
        body.args = form.argsText
          .split(",")
          .map((a) => a.trim())
          .filter(Boolean);
        if (env) body.env = env;
      }
      if (editing) {
        await putJson(`/api/admin/mcp/servers/${editing.id}`, body);
      } else {
        await postJson("/api/admin/mcp/servers", body);
      }
      setFormOpen(false);
      await load();
      notify({
        tone: "success",
        title: editing
          ? (tm.savedToast ?? "Server saved.")
          : (tm.createdToast ?? "Server created."),
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSaving(false);
    }
  };

  const toggleEnabled = async (s: McpServer) => {
    setBusyId(s.id);
    // Optimistic flip; reverted on failure.
    setServers((prev) =>
      prev.map((x) => (x.id === s.id ? { ...x, enabled: !s.enabled } : x)),
    );
    try {
      await putJson(`/api/admin/mcp/servers/${s.id}`, { enabled: !s.enabled });
    } catch (e) {
      setServers((prev) =>
        prev.map((x) => (x.id === s.id ? { ...x, enabled: s.enabled } : x)),
      );
      notify({
        tone: "error",
        title: t.updateFailedTitle ?? "Update failed",
        message: e instanceof Error ? e.message : (t.updateFailed ?? "Update failed"),
      });
    } finally {
      setBusyId(null);
    }
  };

  const remove = async (s: McpServer) => {
    if (
      !confirm(
        tm.deleteConfirm ??
          "Delete this MCP server? Product bindings are detached automatically.",
      )
    )
      return;
    setBusyId(s.id);
    try {
      await del(`/api/admin/mcp/servers/${s.id}`);
      await load();
      notify({ tone: "success", title: tm.deletedToast ?? "Server deleted" });
    } catch (e) {
      notify({
        tone: "error",
        title: t.failed ?? "Failed",
        message: e instanceof Error ? e.message : (t.failed ?? "Failed"),
      });
    } finally {
      setBusyId(null);
    }
  };

  const test = async (s: McpServer) => {
    setTestingId(s.id);
    try {
      const res = (await postJson(
        `/api/admin/mcp/servers/${s.id}/test`,
      )) as McpTestResult;
      setTestView({ server: s, result: res });
      // The backend persists status/status_checked_at/status_error on test.
      await load();
    } catch (e) {
      notify({
        tone: "error",
        title: t.testFailedTitle ?? "Test failed",
        message: e instanceof Error ? e.message : (t.testFailed ?? "Test failed"),
      });
    } finally {
      setTestingId(null);
    }
  };

  const statusLabel = (s: McpServer) =>
    s.status === "ok"
      ? (tm.statusOk ?? "ok")
      : s.status === "error"
        ? (tm.statusError ?? "error")
        : (tm.statusUnknown ?? "unknown");

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tm.loading ?? "Loading MCP servers…"}
      </div>
    );

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">{tm.intro ?? ""}</p>

      <div className="flex items-center justify-end">
        <Button size="sm" onClick={openCreate}>
          <Plus size={14} weight="bold" />
          {tm.addServer ?? "Add server"}
        </Button>
      </div>

      {servers.length === 0 ? (
        <p className="text-sm text-muted">{tm.noServers ?? "No MCP servers yet."}</p>
      ) : (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="px-4 py-3 font-medium">{tm.tableHeaders?.name ?? "Name"}</th>
                <th className="px-4 py-3 font-medium">{tm.tableHeaders?.transport ?? "Transport"}</th>
                <th className="px-4 py-3 font-medium">{tm.tableHeaders?.endpoint ?? "Endpoint"}</th>
                <th className="px-4 py-3 font-medium">{tm.tableHeaders?.enabled ?? "Enabled"}</th>
                <th className="px-4 py-3 font-medium">{tm.tableHeaders?.status ?? "Status"}</th>
                <th className="px-4 py-3 text-right font-medium">{tm.tableHeaders?.actions ?? "Actions"}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-divider">
              {servers.map((s) => {
                const headerKeys = Object.keys(s.headers_masked ?? {});
                const envKeys = Object.keys(s.env_masked ?? {});
                const maskedMeta = [
                  headerKeys.length
                    ? `${tm.headersLabel ?? "headers"}: ${headerKeys.join(", ")}`
                    : null,
                  envKeys.length
                    ? `${tm.envLabel ?? "env"}: ${envKeys.join(", ")}`
                    : null,
                ]
                  .filter(Boolean)
                  .join(" · ");
                return (
                  <tr key={s.id} className="hover:bg-surface-2">
                    <td className="px-4 py-3">
                      <div className="font-medium text-ink">{s.name}</div>
                      {maskedMeta && (
                        <div className="mt-0.5 font-mono text-[11px] text-muted">
                          {maskedMeta}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <Tag tone={s.transport === "http" ? "blue" : "neutral"}>
                        {s.transport}
                      </Tag>
                    </td>
                    <td className="max-w-[280px] px-4 py-3">
                      <div className="truncate font-mono text-xs text-muted">
                        {s.transport === "http"
                          ? (s.url ?? "—")
                          : [s.command, ...(s.args ?? [])].join(" ") || "—"}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <Switch
                        checked={s.enabled}
                        onChange={() => toggleEnabled(s)}
                        disabled={busyId === s.id}
                        label={tm.toggleLabel ?? "Enable or disable this MCP server"}
                      />
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className="inline-flex items-center gap-1.5 text-xs text-muted"
                        title={s.status_error || statusLabel(s)}
                      >
                        <McpStatusDot status={s.status} title={s.status_error || statusLabel(s)} />
                        {statusLabel(s)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="inline-flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => test(s)}
                          disabled={testingId === s.id}
                        >
                          {testingId === s.id ? (
                            <SpinnerIcon />
                          ) : (
                            <Wrench size={14} weight="regular" />
                          )}
                          {tm.test ?? "Test"}
                        </Button>
                        <Button
                          size="sm"
                          variant="subtle"
                          aria-label={tc.edit ?? "Edit"}
                          title={tc.edit ?? "Edit"}
                          className="!px-2"
                          onClick={() => openEdit(s)}
                        >
                          <PencilSimple size={14} weight="regular" />
                        </Button>
                        <Button
                          size="sm"
                          variant="danger"
                          aria-label={tc.delete ?? "Delete"}
                          title={tc.delete ?? "Delete"}
                          className="!px-2"
                          onClick={() => remove(s)}
                          disabled={busyId === s.id}
                        >
                          {busyId === s.id ? (
                            <SpinnerIcon />
                          ) : (
                            <Trash size={14} weight="regular" />
                          )}
                        </Button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}

      {/* Create / edit form */}
      <Modal
        open={formOpen}
        onClose={() => setFormOpen(false)}
        title={editing ? (tm.editServer ?? "Edit server") : (tm.addServer ?? "Add server")}
        size="lg"
        footer={null}
      >
        <form onSubmit={save} className="grid gap-4">
          <div className="grid gap-4 md:grid-cols-[1fr_180px]">
            <div>
              <Label>{tm.name ?? "Name"}</Label>
              <Input
                value={form.name}
                onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))}
                placeholder={tm.namePlaceholder ?? "e.g. Context7 docs"}
                maxLength={128}
                required
                autoFocus
              />
            </div>
            <div>
              <Label>{tm.transport ?? "Transport"}</Label>
              <Select
                value={form.transport}
                onChange={(e) =>
                  setForm((p) => ({ ...p, transport: e.target.value as McpTransport }))
                }
              >
                <option value="http">{tm.transportHttp ?? "HTTP"}</option>
                <option value="stdio">{tm.transportStdio ?? "Stdio"}</option>
              </Select>
            </div>
          </div>

          {form.transport === "http" ? (
            <>
              <div>
                <Label>{tm.url ?? "URL"}</Label>
                <Input
                  type="url"
                  value={form.url}
                  onChange={(e) => setForm((p) => ({ ...p, url: e.target.value }))}
                  placeholder="https://mcp.example.com/mcp"
                  pattern="https?://.+"
                  title="http:// or https:// only"
                  required
                />
              </div>
              <div>
                <Label>{tm.headers ?? "Headers (key=value per line)"}</Label>
                <Textarea
                  value={form.headersText}
                  onChange={(e) => setForm((p) => ({ ...p, headersText: e.target.value }))}
                  placeholder={
                    editing && Object.keys(editing.headers_masked ?? {}).length
                      ? `${tm.secretsStored ?? "stored"}: ${Object.keys(editing.headers_masked ?? {}).join(", ")}`
                      : "Authorization=Bearer …"
                  }
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
                    value={form.command}
                    onChange={(e) => setForm((p) => ({ ...p, command: e.target.value }))}
                    placeholder="npx"
                    required
                  />
                </div>
                <div>
                  <Label>{tm.args ?? "Arguments (comma-separated)"}</Label>
                  <Input
                    value={form.argsText}
                    onChange={(e) => setForm((p) => ({ ...p, argsText: e.target.value }))}
                    placeholder="-y, @modelcontextprotocol/server-everything"
                    className="font-mono text-sm"
                  />
                </div>
              </div>
              <div>
                <Label>{tm.env ?? "Environment (key=value per line)"}</Label>
                <Textarea
                  value={form.envText}
                  onChange={(e) => setForm((p) => ({ ...p, envText: e.target.value }))}
                  placeholder={
                    editing && Object.keys(editing.env_masked ?? {}).length
                      ? `${tm.secretsStored ?? "stored"}: ${Object.keys(editing.env_masked ?? {}).join(", ")}`
                      : "API_KEY=…"
                  }
                  rows={3}
                  spellCheck={false}
                />
              </div>
            </>
          )}

          <div className="flex items-center gap-3 rounded-md border border-divider bg-surface-2 p-3">
            <Switch
              checked={form.enabled}
              onChange={(next) => setForm((p) => ({ ...p, enabled: next }))}
              label={tm.enabled ?? "Enabled"}
            />
            <span className="text-sm text-ink">{tm.enabled ?? "Enabled"}</span>
          </div>

          {editing && (
            <p className="text-xs text-muted">{tm.secretsHint ?? ""}</p>
          )}

          <div className="flex items-center justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => setFormOpen(false)}>
              {tc.cancel ?? "Cancel"}
            </Button>
            <Button type="submit" disabled={saving || !form.name.trim()}>
              {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}
              {tm.save ?? "Save"}
            </Button>
          </div>
        </form>
      </Modal>

      {/* Test result: ok/detail + discovered tools */}
      <Modal
        open={Boolean(testView)}
        onClose={() => setTestView(null)}
        title={
          testView
            ? `${tm.testTitle ?? "Test"} — ${testView.server.name}`
            : (tm.testTitle ?? "Test")
        }
        size="lg"
        footer={null}
      >
        {testView && (
          <div className="space-y-4">
            <Banner tone={testView.result.ok ? "success" : "error"}>
              {testView.result.ok
                ? fmt(tm.testOk ?? "Connected. {n} tool(s) discovered.", {
                    n: testView.result.tools.length,
                  })
                : (testView.result.detail || (tm.testFailed ?? "Connection failed."))}
            </Banner>
            {testView.result.ok && testView.result.detail && (
              <p className="text-sm text-muted">{testView.result.detail}</p>
            )}
            <div>
              <Label>{tm.toolsTitle ?? "Tools"}</Label>
              {testView.result.tools.length === 0 ? (
                <p className="text-sm text-muted">{tm.noTools ?? "No tools discovered."}</p>
              ) : (
                <ul className="flex flex-col gap-2">
                  {testView.result.tools.map((tool) => (
                    <li
                      key={tool.name}
                      className="rounded-md border border-divider bg-surface-2 px-3 py-2"
                    >
                      <div className="font-mono text-sm text-ink">{tool.name}</div>
                      {tool.description && (
                        <p className="mt-0.5 text-xs text-muted">{tool.description}</p>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Rate-limit card (Embedder)                                         */
/* ------------------------------------------------------------------ */

interface RateLimitGroupResponse {
  group: string;
  settings: Record<string, { value: string | null }>;
  resolved?: {
    max_concurrency?: string;
    delay_seconds?: string;
    rate_limit_rps?: string;
  };
}

interface RateLimitValues {
  maxConcurrency: string;
  delaySeconds: string;
  rateLimitRps: string;
}

/**
 * Render the three rate-limit fields (max concurrency / delay / RPS) plus a
 * save button. Used by the Embedder (pgvector memory) rate-limit block inside
 * the merged memory section.
 */
function RateLimitCard({
  labels,
  values,
  onValuesChange,
  onSave,
  saving,
}: {
  labels: Record<string, string>;
  values: RateLimitValues;
  onValuesChange: (patch: Partial<RateLimitValues>) => void;
  onSave: () => void;
  saving: boolean;
}) {
  return (
    <Card className="p-5">
      <div className="grid gap-4 md:grid-cols-3">
        <Field
          label={labels.maxConcurrency ?? "Max Concurrency"}
          value={values.maxConcurrency}
          onChange={(v) => onValuesChange({ maxConcurrency: v })}
          placeholder="2"
        />
        <Field
          label={labels.delaySeconds ?? "Delay Between Requests (sec)"}
          value={values.delaySeconds}
          onChange={(v) => onValuesChange({ delaySeconds: v })}
          placeholder="0.5"
        />
        <Field
          label={labels.rateLimitRps ?? "Rate Limit (RPS)"}
          value={values.rateLimitRps}
          onChange={(v) => onValuesChange({ rateLimitRps: v })}
          placeholder="2.0"
        />
      </div>
      <p className="mt-2 text-xs text-muted">
        {labels.hint ?? ""}
      </p>
      <div className="mt-4 flex items-center gap-2">
        <Button size="sm" onClick={onSave} disabled={saving}>
          {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}
          {labels.save ?? "Save Rate Limits"}
        </Button>
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Memory section (active backend switch + reindex)                   */
/* ------------------------------------------------------------------ */

interface MemoryGroupResponse {
  group: string;
  settings: Record<string, { value: string | null }>;
  resolved?: {
    backend?: string;
    valid_backends?: string[];
    available?: boolean;
    chunk_count?: number;
    product_count?: number;
  };
}

function MemorySection() {
  const { getJson, putJson, postJson, notify } = useAdminApi();
  const { messages } = useLanguage();
  const t = messages?.admin ?? {};
  const tm = t?.memory ?? {};
  const temb = t?.embedder ?? {};

  const [available, setAvailable] = useState<boolean | null>(null);
  const [chunkCount, setChunkCount] = useState<number | null>(null);
  const [productCount, setProductCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [reindexProductId, setReindexProductId] = useState("");

  // Rate-limit state (embedder).
  const [embedderRl, setEmbedderRl] = useState<RateLimitValues>({
    maxConcurrency: "4",
    delaySeconds: "0.1",
    rateLimitRps: "10.0",
  });
  const [savingEmbedderRl, setSavingEmbedderRl] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<MemoryGroupResponse>("/api/admin/memory");
      const r = data?.resolved ?? {};
      setAvailable(typeof r.available === "boolean" ? r.available : null);
      setChunkCount(typeof r.chunk_count === "number" ? r.chunk_count : null);
      setProductCount(typeof r.product_count === "number" ? r.product_count : null);

      // Load the embedder rate-limit settings (a separate setting group on
      // the backend).
      const embedderData = await getJson<RateLimitGroupResponse>("/api/admin/embedder");
      const er = embedderData?.resolved ?? {};
      const es = embedderData?.settings ?? {};
      setEmbedderRl({
        maxConcurrency: es["embedder.max_concurrency"]?.value ?? er.max_concurrency ?? "4",
        delaySeconds: es["embedder.delay_seconds"]?.value ?? er.delay_seconds ?? "0.1",
        rateLimitRps: es["embedder.rate_limit_rps"]?.value ?? er.rate_limit_rps ?? "10.0",
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reindex = async () => {
    setReindexing(true);
    try {
      const body = reindexProductId.trim()
        ? { product_id: reindexProductId.trim() }
        : {};
      const res = (await postJson("/api/admin/memory/reindex", body)) as {
        success?: boolean;
        message?: string;
        reindexed_count?: number;
      };
      const ok = Boolean(res.success);
      notify({
        tone: ok ? "success" : "error",
        title: ok ? (tm.reindexOkTitle ?? "Reindex complete") : (t.failed ?? "Failed"),
        message: res.message || (ok ? (tm.reindexOkMsg ?? "Memory index rebuilt.") : "Reindex failed"),
      });
      if (ok) void load();
    } catch (e) {
      notify({
        tone: "error",
        title: t.failed ?? "Failed",
        message: e instanceof Error ? e.message : "Reindex failed",
      });
    } finally {
      setReindexing(false);
    }
  };

  const saveEmbedderRl = async () => {
    setSavingEmbedderRl(true);
    try {
      const body: Record<string, string> = {
        "embedder.max_concurrency": embedderRl.maxConcurrency.trim(),
        "embedder.delay_seconds": embedderRl.delaySeconds.trim(),
        "embedder.rate_limit_rps": embedderRl.rateLimitRps.trim(),
      };
      await putJson("/api/admin/embedder", body);
      await load();
      notify({ tone: "success", title: temb.savedToast ?? "Saved embedder rate limits." });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSavingEmbedderRl(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tm.loading ?? "Loading memory settings…"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {tm.intro ??
          "Agent memory: the pgvector direct store (Postgres + HNSW cosine recall)."}
      </p>

      <Card className="p-5">
        <SectionHeader
          title={tm.statusHeader ?? "Status"}
          subtitle={tm.statusSub ?? "Live availability + indexed volume for the active backend."}
        />
        <div className="mt-4 flex flex-wrap gap-3">
          <Tag tone={available ? "green" : "neutral"}>
            {available ? (tm.available ?? "available") : (tm.unavailable ?? "unavailable")}
          </Tag>
          {chunkCount !== null && (
            <Tag tone="neutral">
              {(tm.chunks ?? "chunks")}: {chunkCount}
            </Tag>
          )}
          {productCount !== null && (
            <Tag tone="neutral">
              {(tm.products ?? "products")}: {productCount}
            </Tag>
          )}
        </div>
      </Card>

      <Card className="p-5">
        <SectionHeader
          title={temb.title ?? "Embedder rate limits"}
          subtitle={temb.subtitle ?? "Throttles /v1/embeddings calls used by the pgvector memory backend during indexing."}
        />
        <div className="mt-4">
          <RateLimitCard
            labels={temb}
            values={embedderRl}
            onValuesChange={(patch) =>
              setEmbedderRl((prev) => ({ ...prev, ...patch }))
            }
            onSave={saveEmbedderRl}
            saving={savingEmbedderRl}
          />
        </div>
      </Card>

      <Card className="p-5">
        <SectionHeader
          title={tm.reindexHeader ?? "Reindex memory"}
          subtitle={
            tm.reindexSub ??
            "Rebuild the index from source artifacts (codebases, specs, links, knowledge nodes). Leave the product id blank to reindex all products."
          }
        />
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <div className="min-w-[220px] flex-1">
            <Input
              value={reindexProductId}
              onChange={(e) => setReindexProductId(e.target.value)}
              placeholder={tm.productIdPlaceholder ?? "product id (blank = all)"}
            />
          </div>
          <Button size="sm" variant="subtle" onClick={reindex} disabled={reindexing}>
            {reindexing ? <SpinnerIcon /> : <ArrowsCounterClockwise size={14} weight="bold" />}
            {reindexing ? (tm.reindexing ?? "Reindexing…") : (tm.reindexBtn ?? "Reindex")}
          </Button>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Timeouts section (central timeout_config resolvers)                */
/* ------------------------------------------------------------------ */

// Static catalog of every timeout key rendered in the UI. Kept in sync with
// api/timeout_config.TIMEOUT_KEYS; the resolved view from the backend carries
// the effective value / default / floor / unit / group / label, but the list
// of keys + their order + grouping is defined here so the UI renders even
// before the first load (and so the i18n keys are stable).
const TIMEOUT_FIELDS: { key: string; group: string }[] = [
  { key: "llm_request", group: "LLM" },
  { key: "llm_retry_max_time", group: "LLM" },
  { key: "model_list", group: "LLM" },
  { key: "provider_test", group: "LLM" },
  { key: "docgen_indexing_drain", group: "Memory" },
  { key: "memory_query", group: "Memory" },
  { key: "integration_http", group: "Integrations" },
  { key: "git_file_content", group: "Integrations" },
  { key: "mcp_stdio_wait", group: "Integrations" },
  { key: "mermaid_verify", group: "Mermaid" },
  { key: "mermaid_repair", group: "Mermaid" },
  { key: "mermaid_max_repair_attempts", group: "Mermaid" },
];

const TIMEOUT_GROUPS = ["LLM", "Memory", "Integrations", "Mermaid"] as const;

interface TimeoutResolvedEntry {
  value: string;
  default: string;
  floor: string;
  env_var: string;
  label: string;
  unit: string;
  group: string;
}

type TimeoutResolvedView = Record<string, TimeoutResolvedEntry>;

interface TimeoutsGroupResponse {
  group: string;
  settings: Record<string, { value: string | null }>;
  resolved?: TimeoutResolvedView;
}

function TimeoutsSection() {
  const { getJson, putJson, notify } = useAdminApi();
  const { messages } = useLanguage();
  const t = messages?.admin ?? {};
  const tt = t?.timeouts ?? {};

  // One form value per timeout key, keyed by the resolver key (without the
  // "timeouts." prefix). Empty string = no override (fall through to env /
  // default).
  const [values, setValues] = useState<Record<string, string>>({});
  const [resolved, setResolved] = useState<TimeoutResolvedView | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<TimeoutsGroupResponse>("/api/admin/timeouts");
      const r = data?.resolved ?? {};
      setResolved(r);
      const next: Record<string, string> = {};
      for (const f of TIMEOUT_FIELDS) {
        const stored = data?.settings?.[`timeouts.${f.key}`]?.value;
        // Show the stored override if present; otherwise empty (no override).
        next[f.key] = stored ?? "";
      }
      setValues(next);
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const body: Record<string, string> = {};
      for (const f of TIMEOUT_FIELDS) {
        const v = (values[f.key] ?? "").trim();
        // Only send keys the user touched (non-empty). Empty clears: send an
        // explicit empty string so the backend clears the override.
        body[`timeouts.${f.key}`] = v;
      }
      await putJson("/api/admin/timeouts", body);
      await load();
      notify({ tone: "success", title: tt.savedToast ?? "Saved timeout settings." });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSaving(false);
    }
  };

  const groupLabel = (g: string) =>
    tt.groups?.[g] ?? g;

  const fieldLabel = (key: string, fallback: string) =>
    tt.labels?.[key] ?? fallback;

  const fieldHint = (key: string, entry: TimeoutResolvedEntry | undefined) => {
    const floor = entry?.floor ?? "";
    const unit = entry?.unit === "milliseconds" ? "ms" : "s";
    const def = entry?.default ?? "";
    const i18nHint = tt.hints?.[key];
    const parts: string[] = [];
    if (i18nHint) parts.push(i18nHint);
    parts.push(`${tt.floor ?? "floor"} ${floor}${unit}`);
    parts.push(`${tt.default ?? "default"} ${def}${unit}`);
    return parts.join(" · ");
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tt.loading ?? "Loading timeout settings…"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {tt.intro ??
          "Centralized timeout management. Each value is resolved with precedence: admin store > env var > default. Leave a field empty to fall back to the env var or the built-in default."}
      </p>

      {TIMEOUT_GROUPS.map((g) => (
        <Card key={g} className="p-5">
          <SectionHeader title={groupLabel(g)} />
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {TIMEOUT_FIELDS.filter((f) => f.group === g).map((f) => {
              const entry = resolved?.[f.key];
              return (
                <div key={f.key}>
                  <Field
                    label={fieldLabel(f.key, entry?.label ?? f.key)}
                    value={values[f.key] ?? ""}
                    onChange={(v) => setValues((prev) => ({ ...prev, [f.key]: v }))}
                    placeholder={entry?.value ?? entry?.default ?? ""}
                    type="number"
                  />
                  <p className="mt-1 text-xs text-muted">
                    {fieldHint(f.key, entry)}
                  </p>
                </div>
              );
            })}
          </div>
        </Card>
      ))}

      <div className="flex items-center gap-2">
        <Button size="sm" onClick={save} disabled={saving}>
          {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}
          {tt.save ?? "Save"}
        </Button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Prompts section                                                     */
interface PromptFile {
  filename: string;
  size?: number;
  modified?: string;
}

function PromptsSection() {
  const { getJson, putJson, notify } = useAdminApi();
  const { messages, fmt } = useLanguage();
  const t = messages?.admin ?? {};
  const tp = t?.prompts ?? {};
  const [files, setFiles] = useState<PromptFile[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [loadingList, setLoadingList] = useState(true);
  const [loadingFile, setLoadingFile] = useState(false);
  const [saving, setSaving] = useState(false);

  const loadList = useCallback(async () => {
    setLoadingList(true);
    try {
      const data = await getJson<PromptFile[] | { files: PromptFile[] }>(
        "/api/admin/prompts",
      );
      const list = Array.isArray(data) ? data : data?.files ?? [];
      setFiles(list);
      // P2-32: functional update — `selected` must NOT be a dep of loadList,
      // otherwise every selection re-fetches the whole list (double fetch).
      setSelected((prev) =>
        prev && list.some((f) => f.filename === prev)
          ? prev
          : (list[0]?.filename ?? null),
      );
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoadingList(false);
    }
  // P2-32: the list loads once on mount (or explicit refresh); t only labels toasts.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [getJson, notify]);

  useEffect(() => {
    void loadList();
  }, [loadList]);

  const loadFile = useCallback(async (filename: string) => {
    setLoadingFile(true);
    setContent("");
    try {
      // File content endpoint returns {content} or raw text; reuse the shared
      // 401 handling by going through getJson (which redirects on 401).
      const data = await getJson<{ content?: string } | string>(
        `/api/admin/prompts/${encodeURIComponent(filename)}`,
      );
      setContent(typeof data === "string" ? data : data.content ?? "");
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoadingFile(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    if (selected) void loadFile(selected);
  }, [selected, loadFile]);

  const save = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await putJson(`/api/admin/prompts/${encodeURIComponent(selected)}`, { content });
      notify({ tone: "success", title: fmt(tp.savedToast ?? "Saved {file}.", { file: selected }) });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailed ?? "Save failed"),
      });
    } finally {
      setSaving(false);
    }
  };

  if (loadingList) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tp.loadingList ?? "Loading prompts…"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {tp.intro ?? ""}
      </p>
      {files.length === 0 ? (
        <p className="text-sm text-muted">{tp.noFiles ?? "No prompt files found."}</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-[220px_1fr]">
          {/* File list */}
          <aside>
            <nav className="flex flex-col gap-0.5">
              {files.map((f) => (
                <button
                  key={f.filename}
                  onClick={() => setSelected(f.filename)}
                  className={cn(
                    "flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors",
                    selected === f.filename
                      ? "bg-surface-2 font-medium text-ink"
                      : "text-muted hover:bg-surface-2 hover:text-ink",
                  )}
                >
                  <FileText size={14} weight="regular" />
                  <span className="truncate font-mono">{f.filename}</span>
                </button>
              ))}
            </nav>
          </aside>
          {/* Editor */}
          <Card className="p-4">
            {loadingFile ? (
              <div className="flex items-center gap-2 text-sm text-muted">
                <Spinner /> {tp.loadingFile ?? "Loading file…"}
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm text-muted">{selected}</span>
                  <Button size="sm" onClick={save} disabled={saving || !selected}>
                    {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}
                    {tp.save ?? "Save"}
                  </Button>
                </div>
                <Textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  rows={20}
                  spellCheck={false}
                  className="font-mono text-sm"
                />
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Users section                                                       */
/* ------------------------------------------------------------------ */

function UsersSection() {
  const { getJson, putJson, postJson, notify } = useAdminApi();
  const { messages, fmt } = useLanguage();
  const t = messages?.admin ?? {};
  const tu = t?.users ?? {};
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);

  // Create-user form state
  const [nUsername, setNUsername] = useState("");
  const [nEmail, setNEmail] = useState("");
  const [nRole, setNRole] = useState<UserRole>("user");
  const [nPassword, setNPassword] = useState("");
  const [nMustChange, setNMustChange] = useState(true);
  const [creating, setCreating] = useState(false);

  // Revealed credentials (temp password + reset token) shown once.
  const [revealed, setRevealed] = useState<UserCreateResult | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<User[] | { users: User[] }>(
        "/api/admin/users",
      );
      setUsers(Array.isArray(data) ? data : (data?.users ?? []));
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
  }, [load]);

  const setRole = async (u: User, role: UserRole) => {
    setBusy(u.id);
    try {
      // Backend PUT /api/admin/users expects { user_id, role }.
      await putJson("/api/admin/users", { user_id: u.id, role });
      setUsers((prev) => prev.map((x) => (x.id === u.id ? { ...x, role } : x)));
      notify({ tone: "success", title: fmt(tu.roleChangedToast ?? "{name} is now {role}.", { name: u.username, role }) });
    } catch (e) {
      notify({
        tone: "error",
        title: t.updateFailedTitle ?? "Update failed",
        message: e instanceof Error ? e.message : (t.updateFailed ?? "Update failed"),
      });
    } finally {
      setBusy(null);
    }
  };

  const createUser = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!nUsername.trim() || creating) return;
    setCreating(true);
    setRevealed(null);
    try {
      const res = (await postJson("/api/admin/users", {
        username: nUsername.trim(),
        email: nEmail.trim() || undefined,
        role: nRole,
        password: nPassword || undefined,
        must_change_password: nMustChange,
      })) as UserCreateResult;
      setRevealed(res);
      setNUsername("");
      setNEmail("");
      setNPassword("");
      setNRole("user");
      setNMustChange(true);
      await load();
      notify({
        tone: "success",
        title: fmt(tu.createdToastTitle ?? "Created {name}", { name: res.user.username }),
        message: tu.createdToastMessage ?? "",
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.createFailedTitle ?? "Create failed",
        message: e instanceof Error ? e.message : (t.createFailed ?? "Create failed"),
      });
    } finally {
      setCreating(false);
    }
  };

  const issueResetToken = async (u: User) => {
    if (!confirm(fmt(tu.resetTokenConfirm ?? "Issue a new reset token + temp password for {name}?", { name: u.username })))
      return;
    setBusy(u.id);
    setRevealed(null);
    try {
      const res = (await postJson(
        `/api/admin/users/${u.id}/reset-token`,
      )) as UserCreateResult;
      setRevealed(res);
      notify({
        tone: "success",
        title: fmt(tu.resetTokenToastTitle ?? "New reset token for {name}", { name: u.username }),
        message: tu.resetTokenToastMessage ?? "",
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.failed ?? "Failed",
        message: e instanceof Error ? e.message : (t.failed ?? "Failed"),
      });
    } finally {
      setBusy(null);
    }
  };

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tu.loading ?? "Loading users…"}
      </div>
    );

  return (
    <div className="space-y-4">
      {/* Create user form */}
      <Card className="p-5">
        <SectionHeader
          title={tu.createTitle ?? "Create user"}
          subtitle={tu.createSubtitle ?? ""}
        />
        <form onSubmit={createUser} className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <Label>{tu.username ?? "Username"}</Label>
            <Input
              value={nUsername}
              onChange={(e) => setNUsername(e.target.value)}
              placeholder={tu.usernamePlaceholder ?? "username"}
              required
            />
          </div>
          <div>
            <Label>{tu.emailOptional ?? "Email (optional)"}</Label>
            <Input
              type="email"
              value={nEmail}
              onChange={(e) => setNEmail(e.target.value)}
              placeholder={tu.emailPlaceholder ?? "user@example.com"}
            />
          </div>
          <div>
            <Label>{tu.role ?? "Role"}</Label>
            <Select
              value={nRole}
              onChange={(e) => setNRole(e.target.value as UserRole)}
            >
              <option value="user">{tu.roleUser ?? "user"}</option>
              <option value="admin">{tu.roleAdmin ?? "admin"}</option>
            </Select>
          </div>
          <div>
            <Label>{tu.tempPassword ?? "Temp password (optional)"}</Label>
            <Input
              type="password"
              value={nPassword}
              onChange={(e) => setNPassword(e.target.value)}
              placeholder={tu.tempPasswordPlaceholder ?? "leave blank to auto-generate"}
            />
          </div>
          <label className="flex items-center gap-2 text-sm text-muted md:col-span-2">
            <input
              type="checkbox"
              checked={nMustChange}
              onChange={(e) => setNMustChange(e.target.checked)}
              className="h-4 w-4 rounded border-divider"
            />
            {tu.requireChange ?? "Require password change on first login"}
          </label>
          <div className="md:col-span-2 flex justify-end">
            <Button
              type="submit"
              size="sm"
              disabled={creating || !nUsername.trim()}
            >
              {creating ? <SpinnerIcon /> : <Plus size={14} weight="bold" />}
              {tu.createUser ?? "Create user"}
            </Button>
          </div>
        </form>
      </Card>

      {/* Revealed credentials (temp password + reset token) */}
      {revealed && (
        <Card className="border-tag-yellow-bg bg-tag-yellow-bg/40 p-5">
          <SectionHeader
            title={fmt(tu.credentialsTitle ?? "Credentials for {name}", { name: revealed.user.username })}
            subtitle={tu.credentialsSubtitle ?? ""}
          />
          <div className="mt-4 grid gap-3">
            <CredRow label={tu.tempPasswordLabel ?? "Temp password"} value={revealed.temp_password} />
            <CredRow
              label={tu.resetTokenLabel ?? "Reset token"}
              value={revealed.reset_token}
              hint={tu.resetTokenHint ?? ""}
            />
          </div>
        </Card>
      )}

      {/* Users table */}
      <Card className="overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
            <tr>
              <th className="px-4 py-3 font-medium">{tu.tableUser ?? "User"}</th>
              <th className="px-4 py-3 font-medium">{tu.tableProvider ?? "Provider"}</th>
              <th className="px-4 py-3 font-medium">{tu.tableRole ?? "Role"}</th>
              <th className="px-4 py-3 font-medium text-right">{tu.tableActions ?? "Actions"}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-divider">
            {users.map((u) => (
              <tr key={u.id} className="hover:bg-surface-2">
                <td className="px-4 py-3">
                  <div className="font-medium text-ink">{u.username}</div>
                  {u.email && (
                    <div className="text-xs text-muted">{u.email}</div>
                  )}
                  {u.must_change_password && (
                    <div className="mt-0.5 text-[11px] text-tag-yellow-fg">
                      {tu.mustChangePassword ?? "must change password"}
                    </div>
                  )}
                </td>
                <td className="px-4 py-3 text-muted">{u.provider}</td>
                <td className="px-4 py-3">
                  <Tag tone={u.role === "admin" ? "blue" : "neutral"}>
                    {u.role === "admin" ? (tu.roleAdmin ?? "admin") : (tu.roleUser ?? "user")}
                  </Tag>
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="inline-flex items-center gap-2">
                    {u.provider === "local" && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => issueResetToken(u)}
                        disabled={busy === u.id}
                        title={tu.resetTokenTitle ?? ""}
                      >
                        {busy === u.id ? (
                          <SpinnerIcon />
                        ) : (
                          <Key size={14} weight="regular" />
                        )}
                        {tu.resetTokenAction ?? "Reset token"}
                      </Button>
                    )}
                    {u.role === "admin" ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setRole(u, "user")}
                        disabled={busy === u.id}
                      >
                        <UserCircleGear size={14} weight="regular" />
                        {tu.demote ?? "Demote"}
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="subtle"
                        onClick={() => setRole(u, "admin")}
                        disabled={busy === u.id}
                      >
                        <SealCheck size={14} weight="regular" />
                        {tu.promote ?? "Promote"}
                      </Button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {users.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-muted">
                  {tu.noUsers ?? ""}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

/* Copyable credential row (temp password / reset token). */
function CredRow({
  label,
  value,
  hint,
}: {
  label: string;
  value?: string | null;
  hint?: string;
}) {
  const { messages } = useLanguage();
  const tu = messages?.admin?.users ?? {};
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };
  return (
    <div>
      <Label>{label}</Label>
      <div className="flex items-center gap-2">
        <code className="flex-1 truncate rounded-md border border-divider bg-surface-2 px-3 py-2 font-mono text-xs text-ink">
          {value || "—"}
        </code>
        <Button size="sm" variant="subtle" onClick={copy} disabled={!value}>
          <Copy size={14} weight="regular" />
          {copied ? (tu.copied ?? "Copied") : (tu.copy ?? "Copy")}
        </Button>
      </div>
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* API tokens section                                                  */
/* ------------------------------------------------------------------ */

function TokensSection() {
  const { getJson, postJson, del, notify } = useAdminApi();
  const { messages } = useLanguage();
  const t = messages?.admin ?? {};
  const tk = t?.tokens ?? {};
  const [tokens, setTokens] = useState<ApiToken[]>([]);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [revealed, setRevealed] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<ApiToken[] | { tokens: ApiToken[] }>(
        "/api/admin/apitokens",
      );
      setTokens(Array.isArray(data) ? data : (data?.tokens ?? []));
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoading(false);
    }
  }, [getJson, notify, t]);

  useEffect(() => {
    void load();
  }, [load]);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || creating) return;
    setCreating(true);
    try {
      const res = (await postJson("/api/admin/apitokens", {
        name: name.trim(),
      })) as ApiToken;
      setRevealed(res.token ?? null);
      setName("");
      await load();
      notify({
        tone: "success",
        title: tk.createdToastTitle ?? "Token created",
        message: tk.createdToastMessage ?? "",
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.createFailedTitle ?? "Create failed",
        message: e instanceof Error ? e.message : (t.createFailed ?? "Create failed"),
      });
    } finally {
      setCreating(false);
    }
  };

  const revoke = async (id: string) => {
    if (!confirm(tk.revokeConfirm ?? "Revoke this API token?")) return;
    setBusy(id);
    try {
      await del(`/api/admin/apitokens/${id}`);
      setTokens((prev) => prev.filter((tkn) => tkn.id !== id));
      notify({ tone: "success", title: tk.revokedToastTitle ?? "Token revoked" });
    } catch (e) {
      notify({
        tone: "error",
        title: t.revokeFailedTitle ?? "Revoke failed",
        message: e instanceof Error ? e.message : (t.revokeFailed ?? "Revoke failed"),
      });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <form onSubmit={create} className="flex items-end gap-3">
          <div className="flex-1">
            <Label>{tk.tokenName ?? "Token name"}</Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={tk.tokenNamePlaceholder ?? "e.g. CI export token"}
            />
          </div>
          <Button type="submit" disabled={creating || !name.trim()}>
            {creating ? <SpinnerIcon /> : <Plus size={14} weight="bold" />}
            {tk.createToken ?? "Create token"}
          </Button>
        </form>
        {revealed && (
          <div className="mt-4 rounded-md border border-tag-green-bg bg-tag-green-bg px-3 py-2">
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all font-mono text-xs text-tag-green-fg">
                {revealed}
              </code>
              <button
                type="button"
                onClick={() => navigator.clipboard.writeText(revealed)}
                className="inline-flex items-center gap-1 rounded border border-tag-green-fg/30 px-2 py-1 text-xs text-tag-green-fg hover:bg-tag-green-bg"
              >
                <Copy size={12} weight="regular" /> {tk.copy ?? "Copy"}
              </button>
            </div>
          </div>
        )}
      </Card>

      {loading ? (
        <div className="flex items-center gap-2 text-sm text-muted">
          <Spinner /> {tk.loadingTokens ?? "Loading tokens…"}
        </div>
      ) : tokens.length === 0 ? (
        <p className="text-sm text-muted">{tk.noTokens ?? "No API tokens yet."}</p>
      ) : (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="px-4 py-3 font-medium">{tk?.tableHeaders?.name ?? "Name"}</th>
                <th className="px-4 py-3 font-medium">{tk?.tableHeaders?.created ?? "Created"}</th>
                <th className="px-4 py-3 font-medium">{tk?.tableHeaders?.lastUsed ?? "Last used"}</th>
                <th className="px-4 py-3 font-medium text-right">{tk?.tableHeaders?.actions ?? "Actions"}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-divider">
              {tokens.map((tkn) => (
                <tr key={tkn.id} className="hover:bg-surface-2">
                  <td className="px-4 py-3 font-medium text-ink">{tkn.name}</td>
                  <td className="px-4 py-3 text-xs text-muted">
                    {tkn.created_at ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-xs text-muted">
                    {tkn.last_used_at ?? (tk.never ?? "never")}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Button
                      size="sm"
                      variant="danger"
                      onClick={() => revoke(tkn.id)}
                      disabled={busy === tkn.id}
                    >
                      {busy === tkn.id ? (
                        <SpinnerIcon />
                      ) : (
                        <Trash size={14} weight="regular" />
                      )}
                      {tk.revoke ?? "Revoke"}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Admin shell                                                         */
/* ------------------------------------------------------------------ */

function AdminShell() {
  const [section, setSection] = useState<Section>("models");
  const active = SECTIONS.find((s) => s.key === section)!;
  const { messages } = useLanguage();
  const t = messages?.admin ?? {};

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <AppHeader />
      <main className="mx-auto px-6 py-12">
        <SectionHeader
          title={t.title ?? "Admin"}
          subtitle={t.subtitle ?? ""}
        />

        <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[220px_1fr]">
          {/* Section nav */}
          <aside className="lg:sticky lg:top-20 lg:self-start">
            <nav className="flex flex-col gap-0.5">
              {SECTIONS.map((s) => (
                <button
                  key={s.key}
                  onClick={() => setSection(s.key)}
                  className={cn(
                    "flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors",
                    section === s.key
                      ? "bg-surface-2 font-medium text-ink"
                      : "text-muted hover:bg-surface-2 hover:text-ink",
                  )}
                >
                  <s.icon size={16} weight="regular" />
                  {t?.sections?.[s.key] ?? s.key}
                  {section === s.key && (
                    <ArrowRight size={12} weight="bold" className="ml-auto" />
                  )}
                </button>
              ))}
            </nav>
          </aside>

          {/* Section body */}
          <div>
            <h2 className="mb-4 font-editorial text-lg tracking-tight text-ink">
              {t?.sections?.[active.key] ?? active.key}
            </h2>
            {section === "models" && <ModelsSection />}
            {section === "ssl" && <SslSection />}
            {section === "git" && <GitSection />}
            {section === "confluence" && <ConfluenceSection />}
            {section === "integrations" && <IntegrationsSection />}
            {section === "mcp" && <McpSection />}
            {section === "prompts" && <PromptsSection />}
            {section === "memory" && <MemorySection />}
            {section === "timeouts" && <TimeoutsSection />}
            {section === "users" && <UsersSection />}
            {section === "tokens" && <TokensSection />}
          </div>
        </div>
      </main>
    </div>
  );
}

export default function AdminPage() {
  return (
    <AuthGuard requireAdmin>
      <AdminShell />
    </AuthGuard>
  );
}
