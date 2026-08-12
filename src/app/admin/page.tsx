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
  Lightning,
  Plug,
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
  SectionHeader,
  Select,
  Spinner as SpinnerIcon,
  Switch,
  Tag,
  Textarea,
  cn,
} from "@/components/ui";
import type { ApiToken, User, UserCreateResult, UserRole } from "@/lib/types";
import { useLanguage } from "@/contexts/LanguageContext";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

type Section = "models" | "rlm" | "ssl" | "git" | "confluence" | "integrations" | "prompts" | "cognee" | "timeouts" | "users" | "tokens";

const SECTIONS: { key: Section; icon: typeof Gear }[] = [
  { key: "models", icon: Rocket },
  { key: "rlm", icon: Lightning },
  { key: "ssl", icon: Shield },
  { key: "git", icon: GitBranch },
  { key: "confluence", icon: Globe },
  { key: "integrations", icon: Plug },
  { key: "prompts", icon: FileText },
  { key: "cognee", icon: Brain },
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

type ModelTask = "docgen" | "expert" | "summary" | "cognee" | "embedder";

interface ModelCfg {
  model: string;
  base_url: string;
  api_key: string;
  // UI-only: true when a key is already stored (redacted on the server).
  // Never sent on save.
  hasApiKey: boolean;
  // Optional per-model prompt-token budget for RLM (models.<task>.max_prompt_tokens).
  // Empty string = use the default (no override). Only consumed by the docgen
  // RLM path, but surfaced for every task for generality.
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
  "cognee",
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
                placeholder="e.g. qwen3:8b"
              />
              <Field
                label={tm.baseUrl ?? "Base URL"}
                value={c.base_url}
                onChange={(v) => update(task, "base_url", v)}
                placeholder="http://localhost:11434/v1"
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
                    "Optional. RLM prompt-token limit for this model. Leave empty to use the default (200000). Only used by RLM (documentation generation)."}
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
/* RLM / LLM routing section                                           */
/* ------------------------------------------------------------------ */

type RlmTask = "docgen" | "expert" | "summary";
type RlmMode = "auto" | "rlm" | "llm";

interface RlmSettingView {
  value: string | null;
  encrypted: boolean;
  hasKey: boolean;
}
type RlmConfig = {
  group: string;
  settings: Record<string, RlmSettingView>;
  resolved: Record<RlmTask, RlmMode>;
};

const RLM_TASKS: RlmTask[] = ["docgen", "expert", "summary"];

const RLM_MODE_OPTIONS: RlmMode[] = ["auto", "rlm", "llm"];

const RLM_TASK_LABEL_KEY: Record<RlmTask, "taskDocgen" | "taskExpert" | "taskSummary"> = {
  docgen: "taskDocgen",
  expert: "taskExpert",
  summary: "taskSummary",
};
const RLM_TASK_HINT_KEY: Record<RlmTask, "taskDocgenHint" | "taskExpertHint" | "taskSummaryHint"> = {
  docgen: "taskDocgenHint",
  expert: "taskExpertHint",
  summary: "taskSummaryHint",
};
const RLM_MODE_LABEL_KEY: Record<RlmMode, "modeAuto" | "modeRlm" | "modeLlm"> = {
  auto: "modeAuto",
  rlm: "modeRlm",
  llm: "modeLlm",
};
const RLM_MODE_HINT_KEY: Record<RlmMode, "modeAutoHint" | "modeRlmHint" | "modeLlmHint"> = {
  auto: "modeAutoHint",
  rlm: "modeRlmHint",
  llm: "modeLlmHint",
};

function RlmSection() {
  const { getJson, putJson, notify } = useAdminApi();
  const { messages, fmt } = useLanguage();
  const t = messages?.admin ?? {};
  const tr = t?.rlm ?? {};
  const [draft, setDraft] = useState<Record<RlmTask, RlmMode>>({
    docgen: "auto",
    expert: "auto",
    summary: "auto",
  });
  const [resolved, setResolved] = useState<Record<RlmTask, RlmMode> | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<RlmConfig>("/api/admin/rlm");
      setResolved(data.resolved);
      // Seed drafts from the stored value (if any), else from the effective
      // resolved mode so the selector reflects the current routing.
      const next = {} as Record<RlmTask, RlmMode>;
      for (const task of ["docgen", "expert", "summary"] as RlmTask[]) {
        const stored = data.settings[`rlm.${task}.mode`]?.value;
        const v = (stored || data.resolved[task] || "auto").trim().toLowerCase();
        next[task] = v === "rlm" || v === "llm" ? v : "auto";
      }
      setDraft(next);
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

  const save = async () => {
    setSaving(true);
    try {
      const body: Record<string, string> = {};
      for (const task of ["docgen", "expert", "summary"] as RlmTask[]) {
        body[`rlm.${task}.mode`] = draft[task];
      }
      await putJson("/api/admin/rlm", body);
      // Refresh so the resolved tag reflects the post-save effective mode
      // (e.g. forced to "llm" if fast-rlm is not installed).
      await load();
      notify({ tone: "success", title: tr.savedToast ?? "Saved RLM routing modes." });
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
        <Spinner /> {tr.loading ?? "Loading RLM routing…"}
      </div>
    );
  }

  // fast-rlm unavailable: every resolved mode is forced to "llm" regardless of
  // what is stored. Surface this clearly so the admin understands RLM can't run.
  const rlmUnavailable = resolved
    ? (Object.values(resolved) as RlmMode[]).every((m) => m === "llm")
      : false;

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {tr.intro ?? ""}
      </p>
      {rlmUnavailable && (
        <Banner tone="warning">
          {tr.unavailableBanner ?? ""}
        </Banner>
      )}
      {RLM_TASKS.map((taskKey) => {
        const effective = resolved?.[taskKey];
        const draftMode = draft[taskKey];
        const differs = !!effective && effective !== draftMode;
        const label = tr?.[RLM_TASK_LABEL_KEY[taskKey]] ?? taskKey;
        const hint = tr?.[RLM_TASK_HINT_KEY[taskKey]] ?? "";
        return (
          <Card key={taskKey} className="p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="font-editorial text-base tracking-tight text-ink">
                  {label}
                </h3>
                <p className="mt-0.5 text-xs text-muted">{hint}</p>
              </div>
              {effective && (
                <Tag tone={effective === "llm" ? "neutral" : "blue"}>
                  {differs
                    ? fmt(tr.effectiveDiffers ?? "effective: {mode} (draft: {draft})", { mode: effective, draft: draftMode })
                    : fmt(tr.effective ?? "effective: {mode}", { mode: effective })}
                </Tag>
              )}
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <Label>{tr.mode ?? "Mode"}</Label>
                <Select
                  value={draftMode}
                  onChange={(e) =>
                    setDraft((p) => ({
                      ...p,
                      [taskKey]: e.target.value as RlmMode,
                    }))
                  }
                >
                  {RLM_MODE_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>
                      {tr?.[RLM_MODE_LABEL_KEY[opt]] ?? opt} — {tr?.[RLM_MODE_HINT_KEY[opt]] ?? ""}
                    </option>
                  ))}
                </Select>
              </div>
            </div>
          </Card>
        );
      })}
      <div className="flex items-center gap-2">
        <Button size="sm" onClick={save} disabled={saving}>
          {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}
          {tr.save ?? "Save"}
        </Button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Git credentials section                                             */
/* ------------------------------------------------------------------ */

const GIT_HOSTS = ["github", "gitlab"] as const;
type GitHost = (typeof GIT_HOSTS)[number];
interface GitCred {
  url: string;
  token: string;
  // UI-only: true when a token is already stored (redacted on the server).
  hasToken: boolean;
}
type GitConfig = Record<GitHost, GitCred>;

// GET /api/admin/git returns { group, settings, resolved }; the editable
// per-host creds live under `resolved` (token redacted as hasToken).
interface GitGroupResponse {
  group: string;
  settings: Record<string, unknown>;
  resolved: Partial<Record<GitHost, { url: string | null; token: string | null; hasToken: boolean }>>;
}

const DEFAULT_GIT_CRED: GitCred = { url: "", token: "", hasToken: false };

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
        const r = resolved[host];
        next[host] = {
          url: r?.url ?? "",
          token: "",
          hasToken: Boolean(r?.hasToken),
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const update = (host: GitHost, key: keyof GitCred, v: string) =>
    setCfg((prev) => {
      if (!prev) return prev;
      const cur = prev[host] ?? DEFAULT_GIT_CRED;
      return { ...prev, [host]: { ...cur, [key]: v } };
    });

  const save = async (host: GitHost) => {
    if (!cfg) return;
    setSaving(host);
    try {
      const cur = cfg[host] ?? DEFAULT_GIT_CRED;
      // PUT contract expects flat keys `git.<host>.{url,token}`. Only send
      // the token when the admin typed a new value.
      const body: Record<string, string> = { [`git.${host}.url`]: cur.url };
      if (cur.token) body[`git.${host}.token`] = cur.token;
      await putJson("/api/admin/git", body);
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
        const c = cfg?.[host] ?? DEFAULT_GIT_CRED;
        return (
          <Card key={host} className="p-5">
            <h3 className="mb-4 font-editorial text-base tracking-tight text-ink capitalize">
              {host}
            </h3>
            <div className="grid gap-4 md:grid-cols-2">
              <Field
                label={tg.enterpriseUrl ?? "Enterprise / self-hosted URL"}
                value={c.url}
                onChange={(v) => update(host, "url", v)}
                placeholder="https://git.company.com"
              />
              <Field
                label={tg.accessToken ?? "Access token"}
                value={c.token}
                onChange={(v) => update(host, "token", v)}
                placeholder={
                  c.hasToken
                    ? (tg.tokenStored ?? "stored · type to replace")
                    : (tg.tokenPlaceholder ?? "type to set")
                }
                secret
              />
            </div>
            <div className="mt-4 flex items-center gap-2">
              <Button
                size="sm"
                onClick={() => save(host)}
                disabled={saving === host}
              >
                {saving === host ? (
                  <SpinnerIcon />
                ) : (
                  <Gear size={14} weight="regular" />
                )}{" "}
                {tg.save ?? "Save"}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => test(host)}
                disabled={testing === host}
              >
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
/* System prompts section (live-edit refs/prompts/*.md)                */
/* ------------------------------------------------------------------ */
/* Cognee Knowledge Graph section                                      */
/* ------------------------------------------------------------------ */

interface CogneeGroupResponse {
  group: string;
  settings: Record<string, { value: string | null }>;
  resolved?: {
    max_concurrency?: string;
    delay_seconds?: string;
    rate_limit_rps?: string;
  };
}

function CogneeSection() {
  const { getJson, putJson, postJson, notify } = useAdminApi();
  const { messages } = useLanguage();
  const t = messages?.admin ?? {};
  const tcg = (t as Record<string, Record<string, string>>)?.cognee ?? {};

  const [maxConcurrency, setMaxConcurrency] = useState("2");
  const [delaySeconds, setDelaySeconds] = useState("0.5");
  const [rateLimitRps, setRateLimitRps] = useState("2.0");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [reindexing, setReindexing] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJson<CogneeGroupResponse>("/api/admin/cognee");
      const r = data?.resolved ?? {};
      const s = data?.settings ?? {};
      setMaxConcurrency(s["cognee.max_concurrency"]?.value ?? r.max_concurrency ?? "2");
      setDelaySeconds(s["cognee.delay_seconds"]?.value ?? r.delay_seconds ?? "0.5");
      setRateLimitRps(s["cognee.rate_limit_rps"]?.value ?? r.rate_limit_rps ?? "2.0");
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
      const body: Record<string, string> = {
        "cognee.max_concurrency": maxConcurrency.trim(),
        "cognee.delay_seconds": delaySeconds.trim(),
        "cognee.rate_limit_rps": rateLimitRps.trim(),
      };
      await putJson("/api/admin/cognee", body);
      await load();
      notify({ tone: "success", title: tcg.savedToast ?? "Saved Cognee config." });
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

  const forceReindex = async () => {
    setReindexing(true);
    try {
      const res = (await postJson("/api/admin/cognee/reindex", {})) as {
        success?: boolean;
        message?: string;
      };
      const ok = Boolean(res.success);
      notify({
        tone: ok ? "success" : "error",
        title: ok ? (tcg.reindexOkTitle ?? "Reindex Complete") : (t.failed ?? "Failed"),
        message: res.message || (ok ? (tcg.reindexOkMsg ?? "Reindexed Cognee knowledge graph.") : "Reindex failed"),
      });
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

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> {tcg.loading ?? "Loading Cognee settings…"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-[15px] text-muted">
        {tcg.intro ?? "Configure rate limits and concurrency for Cognee knowledge graph indexing to prevent LLM API rate limit overflow."}
      </p>

      <Card className="p-5">
        <div className="grid gap-4 md:grid-cols-3">
          <Field
            label={tcg.maxConcurrency ?? "Max Concurrency"}
            value={maxConcurrency}
            onChange={setMaxConcurrency}
            placeholder="2"
          />
          <Field
            label={tcg.delaySeconds ?? "Delay Between Requests (sec)"}
            value={delaySeconds}
            onChange={setDelaySeconds}
            placeholder="0.5"
          />
          <Field
            label={tcg.rateLimitRps ?? "Rate Limit (RPS)"}
            value={rateLimitRps}
            onChange={setRateLimitRps}
            placeholder="2.0"
          />
        </div>
        <p className="mt-2 text-xs text-muted">
          {tcg.hint ?? "Throttles LLM and embedding requests generated during graph extraction (cognify). Setting max concurrency to 1-3 and delay to 0.5s prevents 429 Too Many Requests errors."}
        </p>
        <div className="mt-4 flex items-center gap-2">
          <Button size="sm" onClick={save} disabled={saving}>
            {saving ? <SpinnerIcon /> : <Gear size={14} weight="regular" />}
            {tcg.save ?? "Save Rate Limits"}
          </Button>
        </div>
      </Card>

      <Card className="p-5">
        <SectionHeader
          title={tcg.reindexHeader ?? "Force Refresh Knowledge Graph"}
          subtitle={tcg.reindexSub ?? "Manually trigger re-indexing of all products and artifacts into the Cognee knowledge graph."}
        />
        <div className="mt-4 flex items-center gap-3">
          <Button size="sm" variant="subtle" onClick={forceReindex} disabled={reindexing}>
            {reindexing ? <SpinnerIcon /> : <ArrowsCounterClockwise size={14} weight="bold" />}
            {reindexing ? (tcg.reindexing ?? "Reindexing...") : (tcg.reindexBtn ?? "Force Refresh Knowledge Graph")}
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
  { key: "cognee_graph_extraction", group: "Cognee" },
  { key: "cognee_cognify", group: "Cognee" },
  { key: "cognee_llm_connection", group: "Cognee" },
  { key: "cognee_init", group: "Cognee" },
  { key: "cognee_recall", group: "Cognee" },
  { key: "docgen_indexing_drain", group: "Cognee" },
  { key: "rlm_api_ms", group: "RLM" },
  { key: "rlm_section", group: "RLM" },
  { key: "rlm_expert", group: "RLM" },
  { key: "integration_http", group: "Integrations" },
  { key: "git_file_content", group: "Integrations" },
  { key: "mcp_stdio_wait", group: "Integrations" },
  { key: "mermaid_verify", group: "Mermaid" },
  { key: "mermaid_repair", group: "Mermaid" },
  { key: "mermaid_max_repair_attempts", group: "Mermaid" },
];

const TIMEOUT_GROUPS = ["LLM", "Cognee", "RLM", "Integrations", "Mermaid"] as const;

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
      if (list.length > 0 && !selected) setSelected(list[0].filename);
    } catch (e) {
      notify({
        tone: "error",
        title: t.loadFailedTitle ?? "Load failed",
        message: e instanceof Error ? e.message : (t.loadFailed ?? "Load failed"),
      });
    } finally {
      setLoadingList(false);
    }
  }, [selected, getJson, notify, t]);

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
            {section === "rlm" && <RlmSection />}
            {section === "ssl" && <SslSection />}
            {section === "git" && <GitSection />}
            {section === "confluence" && <ConfluenceSection />}
            {section === "integrations" && <IntegrationsSection />}
            {section === "prompts" && <PromptsSection />}
            {section === "cognee" && <CogneeSection />}
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
