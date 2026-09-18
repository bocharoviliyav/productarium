"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  Article,
  CaretDown,
  CaretUp,
  Database as DatabaseIcon,
  FileText,
  GitBranch,
  Lightning,
  LinkSimple,
  StopCircle,
  PencilSimple,
  Plus,
  Sparkle,
  Trash,
} from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { ExpertChat } from "@/components/ExpertChat";
import { SummaryBlock } from "@/components/SummaryBlock";
import { KnowledgeTree } from "@/components/knowledge/KnowledgeTree";
import { McpServersPanel } from "@/components/mcp/McpServersPanel";
import { useLanguage } from "@/contexts/LanguageContext";
import {
  Button,
  Card,
  EmptyState,
  IconButton,
  Input,
  Label,
  Modal,
  Reveal,
  SectionHeader,
  Select,
  Spinner,
  Tag,
  cn,
} from "@/components/ui";
import {
  type Codebase,
  type Database,
  type DbPreset,
  type Product,
  type Spec,
  entityPath,
  generateId,
  normalizePages,
  parseLinksContent,
  serializeLinksContent,
} from "@/lib/types";
import { safeExternalHref } from "@/lib/links";
import { useNotifications } from "@/contexts/NotificationContext";

type DeleteType = "codebase" | "spec" | "links" | "database";
type GenerateType = "codebase" | "spec" | "database";

// Progress block reported by the docgen job status/active endpoints
// (api/docgen/jobs.py `_progress_snapshot`).
type DocgenProgress = {
  phase?: string | null;
  sections_total?: number | null;
  sections_done?: number;
  current_section?: string | null;
};

// One in-flight docgen job per entity id. `jobId` null = the initial
// POST /generate is still pending (button lock before a job exists).
type DocgenJob = {
  jobId: string | null;
  type: GenerateType;
  progress: DocgenProgress | null;
};

function withoutKey<T>(record: Record<string, T>, key: string): Record<string, T> {
  const next: Record<string, T> = { ...record };
  delete next[key];
  return next;
}

export default function ProductDetailPage() {
  const router = useRouter();
  const params = useParams<{ productId: string }>();
  const productId = params.productId;

  const [product, setProduct] = useState<Product | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Codebase ("service") add modal — codebase-only, opened from the
  // Codebases section header.
  const [showCodebaseModal, setShowCodebaseModal] = useState(false);
  const [cbName, setCbName] = useState("");
  const [cbRepoUrl, setCbRepoUrl] = useState("");
  const [cbRepoType, setCbRepoType] = useState("github");
  // Write-only access token (P0-2): sent once on create, never read back —
  // the API excludes tokens from responses and exposes `has_token` instead.
  const [cbToken, setCbToken] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  // In-place link add form (inside the Links spoiler).
  const [addLinkOpen, setAddLinkOpen] = useState(false);
  const [linkName, setLinkName] = useState("");
  const [linkUrl, setLinkUrl] = useState("");
  const [linkDesc, setLinkDesc] = useState("");

  // Database ("reverse-engineered") add modal — the hardcoded preset flow.
  // The DSN is typed into a VISIBLE input (the user must be able to check it
  // for typos) and sent exactly once: the backend validates it against the
  // preset registry, proves it with a REAL MCP connection check, then stores
  // it encrypted inside the dedicated preset server row. It is never
  // returned or displayed afterwards — not even in masked form.
  const [showDatabaseModal, setShowDatabaseModal] = useState(false);
  // Preset key (postgresql|mysql|…|oracle) from GET /api/db-presets.
  const [dbType, setDbType] = useState("");
  const [dbName, setDbName] = useState("");
  const [dbDsn, setDbDsn] = useState("");
  // Preset catalog — static, fetched once on mount. null = not loaded yet;
  // [] = endpoint unavailable (the modal shows a hint and blocks submit).
  const [dbPresets, setDbPresets] = useState<DbPreset[] | null>(null);

  // In-flight docgen jobs keyed by entity id — replaces the single
  // `generatingId` so several entities can generate at once and the state
  // survives reloads (restored from GET /docgen/active on mount).
  const [generating, setGenerating] = useState<Record<string, DocgenJob>>({});
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [linksOpen, setLinksOpen] = useState(false);
  // Pending "Generate" confirmation — set by the card buttons, executed by
  // the confirm modal (avoids accidental expensive regenerations).
  const [confirmGen, setConfirmGen] = useState<{ type: GenerateType; entityId: string } | null>(null);

  const { notify } = useNotifications();
  const { messages, fmt } = useLanguage();
  const t = messages?.product ?? {};
  const tc = messages?.common ?? {};
  const tArt = messages?.artifactTypes ?? {};

  const productRef = useRef<Product | null>(null);
  useEffect(() => {
    productRef.current = product;
  }, [product]);

  const generateAbortRef = useRef(false);
  useEffect(() => {
    generateAbortRef.current = false;
    return () => {
      generateAbortRef.current = true;
    };
  }, []);

  const fetchProduct = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/products/${productId}?light=1`, {
        credentials: "include",
        cache: "no-store",
      });
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}`);
        return;
      }
      if (res.status === 404) {
        setError(t.notFound ?? "Product not found.");
        setProduct(null);
        return;
      }
      if (!res.ok) throw new Error(`Failed to load product (${res.status})`);
      setProduct((await res.json()) as Product);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to load product.";
      if (!productRef.current) {
        setError(msg);
      } else {
        notify({ tone: "error", title: t.loadFailedTitle ?? "Product", message: msg });
      }
    } finally {
      setIsLoading(false);
    }
  }, [productId, router, notify]);

  useEffect(() => {
    fetchProduct();
  }, [fetchProduct]);

  // --- Docgen job polling (restorable after reload/navigation) ---------------
  // Poll a 202 job until it settles. Progress snapshots land in `generating`
  // (keyed by entity id, guarded by jobId so a stale poller can never clobber
  // a newer run). Resolves on success (toast + refresh), rejects on
  // failure/timeout, and returns silently once the page unmounts.
  const pollDocgenJob = useCallback(
    async (type: GenerateType, entityId: string, jobId: string) => {
      const maxWaitMs = 30 * 60 * 1000;
      const startedAt = Date.now();
      while (Date.now() - startedAt < maxWaitMs) {
        if (generateAbortRef.current) return;
        await new Promise((r) => setTimeout(r, 2000));
        if (generateAbortRef.current) return;
        // The status routes live under the PLURAL segment — route the
        // segment through entityPath() (see src/lib/types.ts).
        const stRes = await fetch(
          `/api/products/${productId}/${entityPath(type)}/${entityId}/generate/status?job_id=${encodeURIComponent(jobId)}`,
          { credentials: "include", cache: "no-store" },
        );
        if (stRes.status === 404) {
          throw new Error(t.genJobNotFound ?? "Generation job not found.");
        }
        if (!stRes.ok) {
          throw new Error(fmt(t.genStatusFailed, { status: String(stRes.status) }));
        }
        const st = await stRes.json().catch(() => ({}));
        if (st.progress) {
          setGenerating((g) =>
            g[entityId]?.jobId === jobId
              ? { ...g, [entityId]: { ...g[entityId], progress: st.progress } }
              : g,
          );
        }
        if (st.status === "succeeded") {
          notify({
            tone: "success",
            title: t.genTitle ?? "Generation",
            message: st.indexing_message || (t.genDone ?? "Documentation generated."),
          });
          await fetchProduct();
          return;
        }
        if (st.status === "failed") {
          throw new Error(st.error || st.indexing_message || (t.genFailed ?? "Generation failed."));
        }
        // Cancelled by the user: the backend restored the previous version
        // and reindexed — a quiet info toast, not an error.
        if (st.status === "cancelled") {
          notify({
            tone: "info",
            title: t.genTitle ?? "Generation",
            message: st.indexing_message || (t.genCancelled ?? "Generation cancelled."),
          });
          await fetchProduct();
          return;
        }
      }
      throw new Error(t.genTimeout ?? "Generation timed out.");
    },
    [productId, fetchProduct, notify, t, fmt],
  );

  // Poll wrapper with error toast + entry cleanup. Shared by the Generate
  // button and the on-mount restore so both paths notify identically.
  const runDocgenJob = useCallback(
    async (type: GenerateType, entityId: string, jobId: string) => {
      try {
        await pollDocgenJob(type, entityId, jobId);
      } catch (e) {
        if (generateAbortRef.current) return;
        const msg = e instanceof Error ? e.message : (t.genFailed ?? "Generation failed.");
        notify({ tone: "error", title: t.genTitle ?? "Generation", message: msg });
      } finally {
        if (!generateAbortRef.current) {
          // Clear only OUR entry — a newer run may have replaced it.
          setGenerating((g) => (g[entityId]?.jobId === jobId ? withoutKey(g, entityId) : g));
        }
      }
    },
    [pollDocgenJob, notify, t],
  );

  // Latest wrapper without re-triggering the restore effect below.
  const runDocgenJobRef = useRef(runDocgenJob);
  useEffect(() => {
    runDocgenJobRef.current = runDocgenJob;
  }, [runDocgenJob]);

  // Restore in-flight generations after a reload/navigation: the backend
  // registry still tracks them, so the animation + polling resume. Best
  // effort — any failure just leaves the page idle.
  const resumedJobsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch(`/api/products/${productId}/docgen/active`, {
          credentials: "include",
          cache: "no-store",
        });
        if (!res.ok) return;
        const jobs = await res.json();
        if (!Array.isArray(jobs) || cancelled) return;
        for (const job of jobs) {
          const { job_id: jobId, entity_type: type, entity_id: entityId, progress } = job ?? {};
          if (cancelled || !jobId || !entityId) continue;
          if (type !== "codebase" && type !== "spec" && type !== "database") continue;
          // One poller per job even if the effect re-runs (StrictMode/dev).
          if (resumedJobsRef.current.has(jobId)) continue;
          resumedJobsRef.current.add(jobId);
          setGenerating((g) =>
            g[entityId] ? g : { ...g, [entityId]: { jobId, type, progress: progress ?? null } },
          );
          void runDocgenJobRef.current(type, entityId, jobId);
        }
      } catch {
        // Restore is best-effort — ignore network failures.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [productId]);

  // Preset catalog (GET /api/db-presets): powers the database type selector
  // in the add modal and the engine tags on the database cards. Static
  // payload — fetched once on mount; failures degrade to an empty catalog
  // (the modal shows a hint and blocks submit).
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch("/api/db-presets", {
          credentials: "include",
          cache: "no-store",
        });
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        if (!cancelled && Array.isArray(data)) setDbPresets(data as DbPreset[]);
      } catch {
        if (!cancelled) setDbPresets([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Default the type selector to the first preset once the catalog arrives.
  useEffect(() => {
    if (dbPresets?.length && !dbType) setDbType(dbPresets[0].key);
  }, [dbPresets, dbType]);

  const selectedDbPreset = dbPresets?.find((p) => p.key === dbType) ?? null;

  const resetCodebaseForm = () => {
    setCbName("");
    setCbRepoUrl("");
    setCbRepoType("github");
    setCbToken("");
  };

  const resetLinkForm = () => {
    setLinkName("");
    setLinkUrl("");
    setLinkDesc("");
  };

  const resetDatabaseForm = () => {
    setDbName("");
    setDbDsn("");
    // `dbType` deliberately persists across opens — the user's preferred
    // engine is sticky; it is re-defaulted only before the first pick.
  };

  // Add a database via the section-header modal (the hardcoded preset
  // flow): type (from GET /api/db-presets) + visible DSN + name. The POST
  // runs the real MCP connection check server-side and can take a while —
  // the submit button switches to a "checking connection" state.
  const handleAddDatabase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!product || !dbName.trim() || !dbDsn.trim() || !dbType || isSaving) return;
    setIsSaving(true);
    try {
      const body = {
        id: generateId("db"),
        name: dbName.trim(),
        db_type: dbType,
        dsn: dbDsn.trim(),
        source: "manual" as const,
      };
      const res = await fetch(`/api/products/${product.id}/databases?light=1`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Failed to add (${res.status})`);
      }
      setProduct((await res.json()) as Product);
      resetDatabaseForm();
      setShowDatabaseModal(false);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to add.";
      notify({ tone: "error", title: t.addArtifactFailedTitle ?? "Add", message: msg });
    } finally {
      setIsSaving(false);
    }
  };

  // Add a codebase ("service") via the section-header modal. Codebase-only:
  // no type selector, name + git URL + provider.
  const handleAddCodebase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!product || !cbName.trim() || isSaving) return;
    setIsSaving(true);
    try {
      const body = {
        id: generateId("cb"),
        name: cbName.trim(),
        repo_url: cbRepoUrl.trim() || null,
        repo_type: cbRepoType,
        // Empty string would clear a stored token; omit the field entirely
        // when the user left it blank (write-only semantics).
        ...(cbToken.trim() ? { token: cbToken.trim() } : {}),
        source: "manual" as const,
      };
      const res = await fetch(`/api/products/${product.id}/codebases?light=1`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Failed to add (${res.status})`);
      }
      setProduct((await res.json()) as Product);
      resetCodebaseForm();
      setShowCodebaseModal(false);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to add.";
      notify({ tone: "error", title: t.addArtifactFailedTitle ?? "Add", message: msg });
    } finally {
      setIsSaving(false);
    }
  };

  // Add a single link in-place inside the Links spoiler. Creates a Links
  // entity with one {url, description} item; Name + URL + optional desc.
  const handleAddLink = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!product || !linkName.trim() || !linkUrl.trim() || isSaving) return;
    setIsSaving(true);
    try {
      const item = { url: linkUrl.trim(), description: linkDesc.trim() || undefined };
      const body = {
        id: generateId("links"),
        name: linkName.trim(),
        content: serializeLinksContent([item]) || null,
        source: "manual" as const,
      };
      const res = await fetch(`/api/products/${product.id}/links?light=1`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Failed to add (${res.status})`);
      }
      setProduct((await res.json()) as Product);
      resetLinkForm();
      setAddLinkOpen(false);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to add.";
      notify({ tone: "error", title: t.addArtifactFailedTitle ?? "Add", message: msg });
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async (type: DeleteType, entityId: string) => {
    if (!product) return;
    if (!confirm(t.deleteArtifactConfirm ?? "Delete this item?")) return;
    setDeletingId(entityId);
    try {
      // Route the segment through entityPath(): the FastAPI routers register
      // the PLURAL segments (codebases/specs/links/databases) while `type` is
      // the singular form-state value — raw interpolation produces a 404.
      const res = await fetch(
        `/api/products/${product.id}/${entityPath(type)}/${entityId}?light=1`,
        { method: "DELETE", credentials: "include" },
      );
      if (!res.ok) throw new Error(`Failed to delete (${res.status})`);
      setProduct((await res.json()) as Product);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to delete.";
      notify({ tone: "error", title: t.deleteArtifactFailedTitle ?? "Delete", message: msg });
    } finally {
      setDeletingId(null);
    }
  };

  const handleGenerate = async (type: GenerateType, entityId: string) => {
    if (!product || generating[entityId]) return;
    setError(null);
    // Lock the button immediately (also covers the POST itself).
    setGenerating((g) => ({ ...g, [entityId]: { jobId: null, type, progress: null } }));
    let jobId: string | null = null;
    try {
      // The FastAPI generate routes are registered under the PLURAL segment
      // (codebases / specs / databases), but `type` is the singular
      // form-state value. Interpolating it raw produces a 404; route the
      // segment through entityPath() (see src/lib/types.ts).
      const res = await fetch(
        `/api/products/${product.id}/${entityPath(type)}/${entityId}/generate`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          // Generation language is governed by the admin `generation.language`
          // setting (resolved at job start) — no per-request override.
          body: JSON.stringify({}),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Generation failed (${res.status})`);
      }
      jobId = data.job_id ?? null;
      if (!jobId) {
        notify({ tone: "info", title: t.genTitle ?? "Generation", message: data.message || data.status || (t.genTriggered ?? "Generation triggered.") });
        await fetchProduct();
        return;
      }
      setGenerating((g) => ({ ...g, [entityId]: { jobId, type, progress: null } }));
      notify({ tone: "info", title: t.genTitle ?? "Generation", message: t.genStarted ?? "Generation started…" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : (t.genFailed ?? "Generation failed.");
      notify({ tone: "error", title: t.genTitle ?? "Generation", message: msg });
      return;
    } finally {
      // Unlock the pending lock when no polling job took over the entry.
      setGenerating((g) => (g[entityId]?.jobId === null ? withoutKey(g, entityId) : g));
    }
    await runDocgenJob(type, entityId, jobId);
  };

  // Cooperative cancel: flag the job server-side; the poller above observes
  // the terminal "cancelled" status (previous version restored). Idempotent
  // and safe to call while the initial POST /generate is still pending only
  // once a job id exists — the Stop button is disabled until then.
  const handleCancelGenerate = async (type: GenerateType, entityId: string) => {
    if (!product) return;
    try {
      const res = await fetch(
        `/api/products/${product.id}/${entityPath(type)}/${entityId}/generate/cancel`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        },
      );
      if (!res.ok) throw new Error(`Failed to cancel (${res.status})`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : (t.cancelFailed ?? "Failed to cancel.");
      notify({ tone: "error", title: t.genTitle ?? "Generation", message: msg });
    }
  };

  // Card label while generating: "3/7 · System Architecture" in the sections
  // phase, the phase name otherwise (cloning/verifying/…), the plain
  // "Generating…" fallback before the first progress snapshot arrives.
  const genProgressText = (job: DocgenJob): string => {
    const p = job.progress;
    const fallback = t.generating ?? "Generating…";
    if (!p) return fallback;
    const total = p.sections_total ?? 0;
    if (p.phase === "sections" && total > 0) {
      const done = String(Math.max(0, p.sections_done ?? 0));
      if (p.current_section) {
        const sid = String(p.current_section);
        const key = `genSection${sid.charAt(0).toUpperCase()}${sid.slice(1)}`;
        return fmt(t.genProgress ?? "{done}/{total} · {section}", {
          done,
          total: String(total),
          section: t[key] ?? sid,
        });
      }
      return `${done}/${total}`;
    }
    const phaseKey = p.phase
      ? `genPhase${p.phase.charAt(0).toUpperCase()}${p.phase.slice(1)}`
      : null;
    return (phaseKey ? t[phaseKey] : undefined) ?? fallback;
  };

  const onTreeSelect = (node: { node_type: string; id: string }) => {
    if (node.node_type === "page") {
      router.push(`/products/${productId}/knowledge/${node.id}`);
    }
  };

  const codebases = product?.codebases ?? [];
  const specs = product?.specs ?? [];
  const links = product?.links ?? [];
  // Optional on the Product contract (wave E backend lands in parallel) —
  // always read through `?? []` so the page keeps rendering on old payloads.
  const databases = product?.databases ?? [];

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <AppHeader />

      <main className="mx-auto px-6 py-16">
        <Reveal>
          <Link
            href="/"
            className="inline-flex items-center gap-1 text-xs font-medium text-muted transition-colors hover:text-ink"
          >
            <ArrowLeft size={14} weight="bold" />
            {t.allProducts ?? "All products"}
          </Link>
        </Reveal>

        {isLoading ? (
          <div className="mt-12 flex items-center gap-2 text-sm text-muted">
            <Spinner /> {t.loadingProduct ?? "Loading product…"}
          </div>
        ) : error && !product ? (
          <div className="mt-12">
            <EmptyState
              icon={<Article size={20} weight="regular" />}
              title={t.notAvailable ?? "Product not available"}
              description={error}
              action={
                <Button onClick={() => router.push("/")}>
                  {t.backToProducts ?? "Back to products"}
                </Button>
              }
            />
          </div>
        ) : product ? (
          <>
            {/* Header */}
            <Reveal className="mt-6">
              <div className="flex flex-col gap-4 border-b border-divider pb-8 md:flex-row md:items-end md:justify-between">
                <div className="min-w-0">
                  <span className="font-mono text-xs text-muted">
                    {product.id}
                  </span>
                  <h1 className="mt-2 font-editorial text-3xl tracking-tight text-ink">
                    {product.name}
                  </h1>
                  <p className="mt-2 max-w-2xl text-sm text-muted">
                    {product.description || (t.noDescription ?? "No description provided.")}
                  </p>
                </div>
              </div>
            </Reveal>

            {/* Summary block */}
            <Reveal className="mt-8">
              <SummaryBlock product={product} onRefresh={fetchProduct} />
            </Reveal>

            {/* Links — collapsible spoiler under Summary, always rendered so the
                in-place add form is reachable even when there are no links yet. */}
            <Reveal className="mt-6">
              <div className="rounded-lg border border-divider bg-surface">
                <button
                  onClick={() => setLinksOpen((v) => !v)}
                  className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-ink"
                >
                  <span className="inline-flex items-center gap-2">
                    <LinkSimple size={16} weight="regular" />
                    {tArt?.links?.label ?? "Links"}
                    <span className="text-xs text-muted">({links.length})</span>
                  </span>
                  {linksOpen ? <CaretUp size={14} weight="bold" /> : <CaretDown size={14} weight="bold" />}
                </button>
                {linksOpen && (
                  <div className="border-t border-divider px-4 py-3">
                    {links.length === 0 ? (
                      <p className="mb-3 text-xs text-muted">
                        {t.linksEmptyInline ?? "No links yet."}
                      </p>
                    ) : (
                      <ul className="mb-3 flex flex-col gap-3">
                        {links.map((l) => {
                          const items = parseLinksContent(l.content);
                          const firstUrl = items.find((it) => it.url?.trim())?.url;
                          // P0-4: only allowlisted schemes become anchors.
                          const firstHref = safeExternalHref(firstUrl);
                          const isDeleting = deletingId === l.id;
                          return (
                            <li
                              key={l.id}
                              className="rounded-md border border-divider bg-surface-2 p-3"
                            >
                              <div className="flex items-center justify-between gap-2">
                                {firstHref ? (
                                  <a
                                    href={firstHref}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="truncate text-sm font-medium text-ink underline-offset-2 hover:underline"
                                  >
                                    {l.name}
                                  </a>
                                ) : (
                                  <span className="truncate text-sm font-medium text-ink">
                                    {l.name}
                                  </span>
                                )}
                                <div className="flex shrink-0 items-center gap-1.5">
                                  <IconButton
                                    aria-label={tc.edit ?? "Edit"}
                                    title={tc.edit ?? "Edit"}
                                    onClick={() =>
                                      router.push(`/products/${product.id}/artifacts/${l.id}`)
                                    }
                                  >
                                    <PencilSimple size={14} weight="regular" />
                                  </IconButton>
                                  <IconButton
                                    aria-label={t.deleteArtifact ?? "Delete"}
                                    title={t.deleteArtifact ?? "Delete"}
                                    onClick={() => handleDelete("links", l.id)}
                                    disabled={isDeleting}
                                  >
                                    {isDeleting ? <Spinner /> : <Trash size={14} weight="regular" />}
                                  </IconButton>
                                </div>
                              </div>
                              {items.length > 0 && (
                                <ul className="mt-2 flex flex-col gap-1.5">
                                  {items.map((it, i) => {
                                    const itemHref = safeExternalHref(it.url);
                                    return (
                                      <li key={i} className="flex flex-col gap-0.5 text-sm">
                                        {it.url ? (
                                          itemHref ? (
                                            <a
                                              href={itemHref}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              className="font-mono text-xs text-ink underline-offset-2 hover:underline"
                                            >
                                              {it.url}
                                            </a>
                                          ) : (
                                            <span className="font-mono text-xs text-ink">
                                              {it.url}
                                            </span>
                                          )
                                        ) : null}
                                        {it.description && (
                                          <span className="text-muted">{it.description}</span>
                                        )}
                                      </li>
                                    );
                                  })}
                                </ul>
                              )}
                            </li>
                          );
                        })}
                      </ul>
                    )}

                    {/* In-place add form */}
                    {addLinkOpen ? (
                      <form onSubmit={handleAddLink} className="grid gap-2 rounded-md border border-divider bg-surface p-3">
                        <Input
                          value={linkName}
                          onChange={(e) => setLinkName(e.target.value)}
                          placeholder={t.linkName ?? "Link name"}
                          required
                          autoFocus
                        />
                        <Input
                          value={linkUrl}
                          onChange={(e) => setLinkUrl(e.target.value)}
                          placeholder={t.linksUrlPlaceholder ?? "https://…"}
                          required
                        />
                        <Input
                          value={linkDesc}
                          onChange={(e) => setLinkDesc(e.target.value)}
                          placeholder={t.linksDescPlaceholder ?? "Description"}
                        />
                        <div className="flex items-center justify-end gap-2">
                          <Button type="button" variant="ghost" size="sm" onClick={() => setAddLinkOpen(false)}>
                            {tc.cancel ?? "Cancel"}
                          </Button>
                          <Button type="submit" size="sm" disabled={isSaving || !linkName.trim() || !linkUrl.trim()}>
                            {isSaving ? <Spinner /> : <Plus size={14} weight="bold" />}
                            {t.addLink ?? "Add link"}
                          </Button>
                        </div>
                      </form>
                    ) : (
                      <Button type="button" variant="subtle" size="sm" onClick={() => setAddLinkOpen(true)}>
                        <Plus size={14} weight="bold" />
                        {t.addLink ?? "Add link"}
                      </Button>
                    )}
                  </div>
                )}
              </div>
            </Reveal>

            {/* Codebase ("service") add modal — opened from the Codebases section header */}
            <Modal
              open={showCodebaseModal}
              onClose={() => setShowCodebaseModal(false)}
              title={t.addService ?? "Add service"}
              footer={null}
            >
              <form onSubmit={handleAddCodebase} className="grid gap-5">
                <div>
                  <Label>{t.serviceName ?? "Name"}</Label>
                  <Input
                    value={cbName}
                    onChange={(e) => setCbName(e.target.value)}
                    placeholder={t.serviceNamePlaceholder ?? "e.g. Payments API"}
                    required
                    autoFocus
                  />
                </div>
                <div className="grid gap-5 md:grid-cols-[1fr_180px]">
                  <div>
                    <Label>{t.gitUrl ?? "Git URL"}</Label>
                    <Input
                      value={cbRepoUrl}
                      onChange={(e) => setCbRepoUrl(e.target.value)}
                      placeholder={t.gitUrlPlaceholder ?? ""}
                    />
                  </div>
                  <div>
                    <Label>{t.provider ?? "Provider"}</Label>
                    <Select
                      value={cbRepoType}
                      onChange={(e) => setCbRepoType(e.target.value)}
                    >
                      <option value="github">{t.github ?? "GitHub"}</option>
                      <option value="gitlab">{t.gitlab ?? "GitLab"}</option>
                    </Select>
                  </div>
                </div>
                <div>
                  <Label>{t.gitToken ?? "Access token (optional)"}</Label>
                  <Input
                    type="password"
                    value={cbToken}
                    onChange={(e) => setCbToken(e.target.value)}
                    placeholder={
                      t.gitTokenPlaceholder ??
                      "Personal access token — stored encrypted, never shown again"
                    }
                    autoComplete="off"
                  />
                </div>
                <div className="flex items-center justify-end gap-2">
                  <Button type="button" variant="ghost" onClick={() => setShowCodebaseModal(false)}>
                    {tc.cancel ?? "Cancel"}
                  </Button>
                  <Button type="submit" disabled={isSaving || !cbName.trim()}>
                    {isSaving ? <Spinner /> : <Plus size={16} weight="bold" />}
                    {t.saveArtifact ?? "Save"}
                  </Button>
                </div>
              </form>
            </Modal>

            {/* Database add modal — the hardcoded preset flow. Type (from
                GET /api/db-presets) drives the DSN example/hint; the DSN is
                visible (type="text") so typos can be caught, and is sent
                exactly once. */}
            <Modal
              open={showDatabaseModal}
              onClose={() => setShowDatabaseModal(false)}
              title={t.addDatabase ?? "Add database"}
              footer={null}
            >
              <form onSubmit={handleAddDatabase} className="grid gap-5">
                <div>
                  <Label>{t.dbType ?? "Database type"}</Label>
                  <Select
                    value={dbType}
                    onChange={(e) => setDbType(e.target.value)}
                    disabled={!dbPresets?.length}
                  >
                    {(dbPresets ?? []).map((p) => (
                      <option key={p.key} value={p.key}>
                        {p.label}
                      </option>
                    ))}
                  </Select>
                  {dbPresets !== null && dbPresets.length === 0 && (
                    <p className="mt-1.5 text-xs text-muted">
                      {t.dbPresetsFailed ?? ""}
                    </p>
                  )}
                </div>
                <div>
                  <Label>{t.dsn ?? "DSN"}</Label>
                  <Input
                    type="text"
                    value={dbDsn}
                    onChange={(e) => setDbDsn(e.target.value)}
                    placeholder={selectedDbPreset?.dsn_example ?? ""}
                    required
                    autoComplete="off"
                    spellCheck={false}
                    className="font-mono"
                  />
                  {selectedDbPreset?.dsn_hint && (
                    <p className="mt-1.5 font-mono text-xs text-muted">
                      {selectedDbPreset.dsn_hint}
                    </p>
                  )}
                  <p className="mt-1 text-xs text-muted">
                    {t.dsnOnceNote ?? ""}
                  </p>
                </div>
                <div>
                  <Label>{t.databaseName ?? "Database name"}</Label>
                  <Input
                    value={dbName}
                    onChange={(e) => setDbName(e.target.value)}
                    placeholder={t.databaseNamePlaceholder ?? ""}
                    required
                  />
                </div>
                {selectedDbPreset && (
                  <p className="text-xs text-muted">
                    {t.dbPresetPoweredBy ?? "Built-in MCP server:"}{" "}
                    <a
                      href={selectedDbPreset.server.homepage}
                      target="_blank"
                      rel="noreferrer"
                      className="underline decoration-divider underline-offset-2 transition-colors hover:text-ink"
                    >
                      {selectedDbPreset.server.name}
                    </a>{" "}
                    · {selectedDbPreset.server.license}
                  </p>
                )}
                <div className="flex items-center justify-end gap-2">
                  <Button type="button" variant="ghost" onClick={() => setShowDatabaseModal(false)}>
                    {tc.cancel ?? "Cancel"}
                  </Button>
                  <Button
                    type="submit"
                    disabled={isSaving || !dbType || !dbName.trim() || !dbDsn.trim()}
                  >
                    {isSaving ? <Spinner /> : <Plus size={16} weight="bold" />}
                    {isSaving
                      ? (t.dbChecking ?? "Checking connection…")
                      : (t.saveArtifact ?? "Save")}
                  </Button>
                </div>
              </form>
            </Modal>

            {/* Generate confirmation — new immutable version on every run */}
            <Modal
              open={Boolean(confirmGen)}
              onClose={() => setConfirmGen(null)}
              title={t.confirmGenTitle ?? "Generate documentation?"}
              footer={null}
            >
              <p className="text-sm text-muted">
                {t.confirmGenText ?? "A new documentation version will be generated."}
              </p>
              <div className="mt-6 flex items-center justify-end gap-2">
                <Button variant="ghost" onClick={() => setConfirmGen(null)}>
                  {tc.cancel ?? "Cancel"}
                </Button>
                <Button
                  onClick={() => {
                    const g = confirmGen;
                    setConfirmGen(null);
                    if (g) void handleGenerate(g.type, g.entityId);
                  }}
                >
                  <Lightning size={16} weight="fill" />
                  {t.generate ?? "Generate"}
                </Button>
              </div>
            </Modal>

            {/* Two-column: (specs + knowledge tree) | main content */}
            <div className="mt-12 grid grid-cols-1 gap-8 lg:grid-cols-[320px_1fr]">
              <aside className="lg:sticky lg:top-20 lg:self-start">
                {/* Specs — above the knowledge tree, styled like the knowledge tree */}
                <div className="mb-3">
                  <Card className="p-3">
                    <div className="flex items-center justify-between px-1 pb-2">
                      <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
                        {tArt?.spec?.label ?? "Specs"}
                      </h3>
                      <IconButton
                        aria-label={t.addSpec ?? "Add spec"}
                        title={t.addSpec ?? "Add spec"}
                        className="h-7 w-7"
                        onClick={() => router.push(`/products/${product.id}/specs/new`)}
                      >
                        <Plus size={14} weight="bold" />
                      </IconButton>
                    </div>
                    {specs.length === 0 ? (
                      <div className="rounded-md border border-dashed border-divider bg-surface-2 px-3 py-6 text-center text-xs text-muted">
                        {t.specEmptyDesc ?? ""}
                      </div>
                    ) : (
                      <nav className="flex flex-col gap-0.5">
                        {specs.map((s: Spec) => (
                          <button
                            key={s.id}
                            onClick={() =>
                              router.push(`/products/${product.id}/specs/${s.id}`)
                            }
                            className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-sm text-muted transition-colors hover:bg-surface-2 hover:text-ink"
                          >
                            <span className="shrink-0 text-muted">
                              <FileText size={14} weight="regular" />
                            </span>
                            <span className="min-w-0 flex-1 truncate">{s.name}</span>
                            <Tag tone="green">{s.kind}</Tag>
                          </button>
                        ))}
                      </nav>
                    )}
                  </Card>
                </div>

                <Card className="p-3">
                  <KnowledgeTree
                    productId={product.id}
                    onSelect={onTreeSelect}
                    onMutate={fetchProduct}
                  />
                </Card>
              </aside>

              <div className="flex flex-col gap-12">
                {/* Codebases */}
                <section>
                  <SectionHeader
                    title={tArt?.codebase?.label ?? "Codebases"}
                    subtitle={
                      codebases.length
                        ? fmt(t.artifactsCount, { n: codebases.length })
                        : (t.artifactsEmptySubtitle ?? "")
                    }
                    action={
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={() => setShowCodebaseModal(true)}
                      >
                        <Plus size={14} weight="bold" />
                        {t.addService ?? "Add service"}
                      </Button>
                    }
                  />

                  <div className="mt-6">
                    {codebases.length === 0 ? (
                      <EmptyState
                        icon={<Plus size={20} weight="regular" />}
                        title={t.noArtifactsTitle ?? "No codebases yet"}
                        description={t.noArtifactsDesc ?? ""}
                        action={
                          <Button onClick={() => setShowCodebaseModal(true)}>
                            <Plus size={16} weight="bold" />
                            {t.addService ?? "Add service"}
                          </Button>
                        }
                      />
                    ) : (
                      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                        {codebases.map((c: Codebase, i) => {
                          const gen = generating[c.id] ?? null;
                          const isGenerating = Boolean(gen);
                          const isDeleting = deletingId === c.id;
                          // Light payloads strip generated_docs when pages
                          // exist — metadata presence is the fallback signal.
                          const hasDocs =
                            Boolean(c.generated_docs) || normalizePages(c.pages).length > 0;
                          return (
                            <Reveal key={c.id} delayMs={Math.min(i, 6) * 80}>
                              <Card
                                hover
                                className="group relative flex h-full flex-col overflow-hidden p-6"
                              >
                                {isGenerating && (
                                  <span className="gen-progress-bar" aria-hidden />
                                )}
                                <div className="flex items-start justify-between gap-3">
                                  <div className="flex items-center gap-2.5">
                                    <span className="flex h-9 w-9 items-center justify-center rounded-md bg-surface-2 text-ink">
                                      <GitBranch size={18} weight="regular" />
                                    </span>
                                    <div className="min-w-0">
                                      <h3 className="truncate text-sm font-medium text-ink">
                                        {c.name}
                                      </h3>
                                      <div className="mt-1 flex flex-wrap items-center gap-2">
                                        <Tag tone="blue">{tArt?.codebase?.label ?? "Codebase"}</Tag>
                                        {hasDocs && (
                                          <Tag tone="green">{t.docsReady ?? "Docs ready"}</Tag>
                                        )}
                                        {c.has_token && (
                                          <Tag tone="neutral">{t.tokenSaved ?? "Token saved"}</Tag>
                                        )}
                                        {c.verified && (
                                          <Tag tone="green">{t.verified ?? "Verified"}</Tag>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                  <IconButton
                                    aria-label={t.deleteArtifact ?? "Delete"}
                                    title={t.deleteArtifact ?? "Delete"}
                                    onClick={() => handleDelete("codebase", c.id)}
                                    disabled={isDeleting}
                                    className="opacity-0 transition-opacity group-hover:opacity-100"
                                  >
                                    {isDeleting ? <Spinner /> : <Trash size={16} weight="regular" />}
                                  </IconButton>
                                </div>

                                {isGenerating && (
                                  <p className="mt-4 flex items-center gap-2 font-mono text-xs text-muted">
                                    <Spinner className="h-3.5 w-3.5" />
                                    {genProgressText(gen)}
                                  </p>
                                )}

                                {c.repo_url && (
                                  <p className="mt-4 truncate font-mono text-xs text-muted">
                                    {c.repo_url}
                                  </p>
                                )}

                                {c.generated_docs && (
                                  <div className="mt-4 max-h-28 overflow-hidden rounded-md border border-divider bg-surface-2 p-3 font-mono text-xs leading-relaxed text-muted">
                                    {c.generated_docs.slice(0, 280)}
                                    {c.generated_docs.length > 280 && "…"}
                                  </div>
                                )}

                                <div className="mt-6 flex items-center justify-between border-t border-divider pt-4">
                                  <button
                                    onClick={() =>
                                      router.push(`/products/${product.id}/artifacts/${c.id}`)
                                    }
                                    className={cn(
                                      "inline-flex items-center gap-1 text-xs font-medium text-ink",
                                      "transition-transform hover:translate-x-0.5",
                                    )}
                                  >
                                    {t.openDocs ?? "Open"}
                                    <ArrowRight size={14} weight="bold" />
                                  </button>
                                  {isGenerating ? (
                                    <Button
                                      size="sm"
                                      variant="subtle"
                                      onClick={() => handleCancelGenerate("codebase", c.id)}
                                      disabled={!gen?.jobId}
                                    >
                                      <StopCircle size={14} weight="fill" />
                                      {t.stopGeneration ?? "Stop"}
                                    </Button>
                                  ) : (
                                    <Button
                                      size="sm"
                                      variant="subtle"
                                      onClick={() => setConfirmGen({ type: "codebase", entityId: c.id })}
                                    >
                                      <Lightning size={14} weight="fill" />
                                      {t.generate ?? "Generate"}
                                    </Button>
                                  )}
                                </div>
                              </Card>
                            </Reveal>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </section>

                {/* Databases (reverse-engineered via MCP tools) */}
                <section>
                  <SectionHeader
                    title={tArt?.database?.label ?? "Databases"}
                    subtitle={
                      databases.length
                        ? fmt(t.databasesCount ?? "{n} database(s)", { n: databases.length })
                        : (t.dbEmptySubtitle ?? "")
                    }
                    action={
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={() => setShowDatabaseModal(true)}
                      >
                        <Plus size={14} weight="bold" />
                        {t.addDatabase ?? "Add database"}
                      </Button>
                    }
                  />

                  <div className="mt-6">
                    {databases.length === 0 ? (
                      <EmptyState
                        icon={<DatabaseIcon size={20} weight="regular" />}
                        title={t.noDatabasesTitle ?? "No databases yet"}
                        description={t.noDatabasesDesc ?? ""}
                        action={
                          <Button onClick={() => setShowDatabaseModal(true)}>
                            <Plus size={16} weight="bold" />
                            {t.addDatabase ?? "Add database"}
                          </Button>
                        }
                      />
                    ) : (
                      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                        {databases.map((d: Database, i) => {
                          const gen = generating[d.id] ?? null;
                          const isGenerating = Boolean(gen);
                          const isDeleting = deletingId === d.id;
                          const hasDocs =
                            Boolean(d.generated_docs) || normalizePages(d.pages).length > 0;
                          return (
                            <Reveal key={d.id} delayMs={Math.min(i, 6) * 80}>
                              <Card
                                hover
                                className="group relative flex h-full flex-col overflow-hidden p-6"
                              >
                                {isGenerating && (
                                  <span className="gen-progress-bar" aria-hidden />
                                )}
                                <div className="flex items-start justify-between gap-3">
                                  <div className="flex items-center gap-2.5">
                                    <span className="flex h-9 w-9 items-center justify-center rounded-md bg-surface-2 text-ink">
                                      <DatabaseIcon size={18} weight="regular" />
                                    </span>
                                    <div className="min-w-0">
                                      <h3 className="truncate text-sm font-medium text-ink">
                                        {d.name}
                                      </h3>
                                      <div className="mt-1 flex flex-wrap items-center gap-2">
                                        <Tag tone="blue">{tArt?.database?.label ?? "Database"}</Tag>
                                        {d.db_type && (
                                          <Tag tone="neutral">
                                            {dbPresets?.find((p) => p.key === d.db_type)?.label ?? d.db_type}
                                          </Tag>
                                        )}
                                        {d.mcp_server_name && (
                                          <Tag tone="neutral">{d.mcp_server_name}</Tag>
                                        )}
                                        {hasDocs && (
                                          <Tag tone="green">{t.docsReady ?? "Docs ready"}</Tag>
                                        )}
                                        {d.verified && (
                                          <Tag tone="green">{t.verified ?? "Verified"}</Tag>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                  <IconButton
                                    aria-label={t.deleteArtifact ?? "Delete"}
                                    title={t.deleteArtifact ?? "Delete"}
                                    onClick={() => handleDelete("database", d.id)}
                                    disabled={isDeleting}
                                    className="opacity-0 transition-opacity group-hover:opacity-100"
                                  >
                                    {isDeleting ? <Spinner /> : <Trash size={16} weight="regular" />}
                                  </IconButton>
                                </div>

                                {isGenerating && (
                                  <p className="mt-4 flex items-center gap-2 font-mono text-xs text-muted">
                                    <Spinner className="h-3.5 w-3.5" />
                                    {genProgressText(gen)}
                                  </p>
                                )}

                                {d.generated_docs && (
                                  <div className="mt-4 max-h-28 overflow-hidden rounded-md border border-divider bg-surface-2 p-3 font-mono text-xs leading-relaxed text-muted">
                                    {d.generated_docs.slice(0, 280)}
                                    {d.generated_docs.length > 280 && "…"}
                                  </div>
                                )}

                                <div className="mt-6 flex items-center justify-between border-t border-divider pt-4">
                                  <button
                                    onClick={() =>
                                      router.push(`/products/${product.id}/artifacts/${d.id}`)
                                    }
                                    className={cn(
                                      "inline-flex items-center gap-1 text-xs font-medium text-ink",
                                      "transition-transform hover:translate-x-0.5",
                                    )}
                                  >
                                    {t.openDocs ?? "Open"}
                                    <ArrowRight size={14} weight="bold" />
                                  </button>
                                  {isGenerating ? (
                                    <Button
                                      size="sm"
                                      variant="subtle"
                                      onClick={() => handleCancelGenerate("database", d.id)}
                                      disabled={!gen?.jobId}
                                    >
                                      <StopCircle size={14} weight="fill" />
                                      {t.stopGeneration ?? "Stop"}
                                    </Button>
                                  ) : (
                                    <Button
                                      size="sm"
                                      variant="subtle"
                                      onClick={() => setConfirmGen({ type: "database", entityId: d.id })}
                                    >
                                      <Lightning size={14} weight="fill" />
                                      {t.generate ?? "Generate"}
                                    </Button>
                                  )}
                                </div>
                              </Card>
                            </Reveal>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </section>

                {/* Expert agent chat */}
                <section>
                  <Reveal>
                    <Card className="p-6 md:p-8">
                      <SectionHeader
                        title={t.askExpertTitle ?? "Ask expert"}
                        subtitle={t.askExpertSubtitle ?? ""}
                        action={
                          <span className="inline-flex items-center gap-1.5 rounded-full bg-tag-yellow-bg px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wide text-tag-yellow-fg">
                            <Sparkle size={12} weight="fill" />
                            {t.expertBadge ?? "expert"}
                          </span>
                        }
                      />
                      <div className="mt-6">
                        <ExpertChat productId={product.id} />
                      </div>
                    </Card>
                  </Reveal>
                </section>

                {/* MCP servers bound to this product (tools for the expert agent) */}
                <Reveal>
                  <McpServersPanel productId={product.id} />
                </Reveal>
              </div>
            </div>
          </>
        ) : null}
      </main>
    </div>
  );
}
