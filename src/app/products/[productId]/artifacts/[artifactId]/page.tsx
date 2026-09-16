"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowCounterClockwise,
  ArrowLeft,
  CaretDown,
  Database as DatabaseIcon,
  FileText,
  GitBranch,
  Lightning,
  LinkSimple,
  PencilSimple,
  Plus,
  SealCheck,
  StopCircle,
  Trash,
} from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { VerifiedBadge, VerifiedButton } from "@/components/VerifiedBadge";
import { MarkdownEditor, EditorSaveBar } from "@/components/MarkdownEditor";
import { SpecViewer } from "@/components/SpecViewer";
import { LinksViewer } from "@/components/LinksViewer";
import { ProvenancePanel } from "@/components/ProvenancePanel";
import dynamic from "next/dynamic";
import {
  Button,
  Card,
  EmptyState,
  IconButton,
  Input,
  Modal,
  Reveal,
  SectionHeader,
  Select,
  Spinner,
  Tag,
  Textarea,
  cn,
} from "@/components/ui";
import {
  type ArtifactPage,
  type Codebase,
  type Database,
  type DocVersionDetail,
  type DocVersionList,
  type EntityKind,
  type LinkItem,
  type Links,
  type Product,
  type Spec,
  entityPath,
  normalizePages,
  parseLinksContent,
  serializeLinksContent,
} from "@/lib/types";

const Markdown = dynamic(() => import("@/components/Markdown"), {
  ssr: false,
  loading: () => <div className="text-sm text-muted">{"Loading…"}</div>,
});

export default function EntityDocsViewer() {
  const params = useParams<{ productId: string; artifactId: string }>();
  const { productId, artifactId } = params;
  const router = useRouter();
  const { notify } = useNotifications();
  const { messages, fmt } = useLanguage();
  const t = messages?.artifact ?? {};
  const tc = messages?.common ?? {};
  const tArt = messages?.artifactTypes ?? {};

  const [product, setProduct] = useState<Product | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activePageId, setActivePageId] = useState<string | null>(null);
  const [verified, setVerified] = useState(false);
  const [verifiedBy, setVerifiedBy] = useState<string | null>(null);

  const [editing, setEditing] = useState(false);
  const [draftContent, setDraftContent] = useState("");
  const [draftLinks, setDraftLinks] = useState<LinkItem[]>([]);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Doc versions (immutable history): list + selected archived version.
  const [versions, setVersions] = useState<DocVersionList | null>(null);
  const [viewVersion, setViewVersion] = useState<number | null>(null);
  const [versionDetail, setVersionDetail] = useState<DocVersionDetail | null>(null);
  // Per-page regeneration: confirm modal + locally polled job.
  const [confirmRegen, setConfirmRegen] = useState(false);
  const [regenJob, setRegenJob] = useState<string | null>(null);
  const [regenStarting, setRegenStarting] = useState(false);
  // Rollback confirm + in-flight restore.
  const [confirmRestore, setConfirmRestore] = useState<number | null>(null);
  const [restoring, setRestoring] = useState(false);

  // Silences poller toasts once the viewer unmounts.
  const viewerAbortRef = useRef(false);
  useEffect(() => {
    viewerAbortRef.current = false;
    return () => {
      viewerAbortRef.current = true;
    };
  }, []);

  const fetchProduct = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/products/${productId}`, {
        credentials: "include",
        cache: "no-store",
      });
      if (res.status === 401) {
        router.replace(
          `/login?next=/products/${productId}/artifacts/${artifactId}`,
        );
        return;
      }
      if (res.status === 404) {
        setError(t.productNotFound ?? "Product not found.");
        return;
      }
      if (!res.ok) throw new Error(`Failed to load (${res.status})`);
      const data = (await res.json()) as Product;
      setProduct(data);
      const found = findEntity(data, artifactId);
      setVerified(Boolean(found?.entity?.verified));
      setVerifiedBy(found?.entity?.verified_by ?? null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to load.";
      setError(msg);
      notify({ tone: "error", title: t.loadFailedTitle ?? "Failed to load", message: msg });
    } finally {
      setIsLoading(false);
    }
  }, [productId, artifactId, router, notify]);

  useEffect(() => {
    fetchProduct();
  }, [fetchProduct]);

  const { entity, kind } = useMemo(() => {
    if (!product) return { entity: undefined, kind: undefined as EntityKind | undefined };
    return findEntity(product, artifactId);
  }, [product, artifactId]);

  // Specs now have a dedicated editor route; redirect any spec deep link away
  // from the generic artifact viewer to /products/{id}/specs/{specId}.
  useEffect(() => {
    if (kind === "spec" && artifactId) {
      router.replace(`/products/${productId}/specs/${artifactId}`);
    }
  }, [kind, artifactId, productId, router]);

  // Databases share the codebase page-tree contract (pages dict keyed by id,
  // plus the additive provenance block the verification pipeline persists).
  const pages: ArtifactPage[] = useMemo(
    () =>
      entity && (kind === "codebase" || kind === "database")
        ? normalizePages((entity as Codebase | Database).pages)
        : [],
    [entity, kind],
  );

  // Version history (codebase/database): best-effort list — a failure just
  // leaves the selector hidden.
  const fetchVersions = useCallback(async () => {
    if (kind !== "codebase" && kind !== "database") return;
    try {
      const res = await fetch(
        `/api/products/${productId}/${entityPath(kind)}/${artifactId}/versions`,
        { credentials: "include", cache: "no-store" },
      );
      if (!res.ok) return;
      setVersions((await res.json()) as DocVersionList);
    } catch {
      /* best-effort */
    }
  }, [productId, artifactId, kind]);

  useEffect(() => {
    setVersions(null);
    setViewVersion(null);
    setVersionDetail(null);
    fetchVersions();
  }, [fetchVersions]);

  // Archived snapshot (read-only): fetched on select, dropped on deselect.
  useEffect(() => {
    if (viewVersion === null || !kind) {
      setVersionDetail(null);
      return;
    }
    let cancelled = false;
    setVersionDetail(null);
    void (async () => {
      try {
        const res = await fetch(
          `/api/products/${productId}/${entityPath(kind)}/${artifactId}/versions/${viewVersion}`,
          { credentials: "include", cache: "no-store" },
        );
        if (!res.ok) throw new Error(String(res.status));
        const d = (await res.json()) as DocVersionDetail;
        if (!cancelled) setVersionDetail(d);
      } catch {
        if (!cancelled) {
          notify({
            tone: "error",
            title: t.versionsTitle ?? "Versions",
            message: t.versionsLoadFailed ?? "Failed to load version.",
          });
          setViewVersion(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [viewVersion, productId, artifactId, kind, notify]);

  const archivePages = useMemo(
    () => (versionDetail ? normalizePages(versionDetail.pages) : []),
    [versionDetail],
  );
  const viewingArchive = viewVersion !== null;
  const displayPages = viewingArchive ? archivePages : pages;

  // Keep the active page valid across version switches: reset to the first
  // page whenever the selection disappears from the display set.
  useEffect(() => {
    if (displayPages.length > 0 && !displayPages.some((p) => p.id === activePageId)) {
      setActivePageId(displayPages[0].id);
    }
  }, [displayPages, activePageId]);

  const activePage = useMemo(
    () => displayPages.find((p) => p.id === activePageId) ?? null,
    [displayPages, activePageId],
  );

  // Nested page tree: a page whose `parent` references another page in the
  // set renders as its child (insertion order, no alphabetical sorting —
  // matches the backend's canonical section order). Legacy flat pages and
  // orphaned `parent` references render top-level exactly as before.
  const pageTree = useMemo(() => {
    const ids = new Set(displayPages.map((p) => p.id));
    const childrenByParent = new Map<string, ArtifactPage[]>();
    const roots: ArtifactPage[] = [];
    for (const p of displayPages) {
      const parent = p.parent?.trim();
      if (parent && parent !== p.id && ids.has(parent)) {
        const list = childrenByParent.get(parent) ?? [];
        list.push(p);
        childrenByParent.set(parent, list);
      } else {
        roots.push(p);
      }
    }
    return { roots, childrenByParent };
  }, [displayPages]);

  // Collapsible nav groups: with hundreds of per-table subpages the nav is
  // unreadable, so children stay hidden until the group opens; the group
  // holding the active page auto-opens so deep links remain navigable.
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const activeParent = activePage?.parent?.trim();
  useEffect(() => {
    if (activeParent) {
      setOpenGroups((g) => (g[activeParent] ? g : { ...g, [activeParent]: true }));
    }
  }, [activeParent]);

  const isCodebase = kind === "codebase";
  const isDatabase = kind === "database";
  const isSpec = kind === "spec";
  const isLinks = kind === "links";
  const codebase = entity as Codebase | undefined;
  const databaseEntity = entity as Database | undefined;
  const spec = entity as Spec | undefined;
  const linksEntity = entity as Links | undefined;

  const hasDocs = Boolean(
    codebase?.generated_docs || databaseEntity?.generated_docs || pages.length > 0,
  );
  const hasRawContent = Boolean(spec?.content || linksEntity?.content);
  // Archived versions are read-only — editing targets the current version.
  const canEdit =
    !viewingArchive &&
    ((isCodebase && hasDocs) ||
      (isDatabase && hasDocs) ||
      ((isSpec || isLinks) && hasRawContent));

  useEffect(() => {
    if (editing) return;
    if (isLinks) {
      setDraftLinks(
        parseLinksContent(linksEntity?.content).length
          ? parseLinksContent(linksEntity?.content)
          : [{ url: "", description: "" }],
      );
    } else {
      setDraftContent(
        activePage
          ? activePage.content || ""
          : spec?.content || codebase?.generated_docs || databaseEntity?.generated_docs || "",
      );
    }
    setDirty(false);
  }, [activePage, codebase, databaseEntity, spec, linksEntity, editing, isLinks]);

  const startEditing = () => {
    if (isLinks) {
      setDraftLinks(
        parseLinksContent(linksEntity?.content).length
          ? parseLinksContent(linksEntity?.content)
          : [{ url: "", description: "" }],
      );
    } else {
      setDraftContent(
        activePage
          ? activePage.content || ""
          : spec?.content || codebase?.generated_docs || databaseEntity?.generated_docs || "",
      );
    }
    setDirty(false);
    setEditing(true);
  };

  const save = async () => {
    if (!kind || saving) return;
    setSaving(true);
    try {
      let payload: Record<string, unknown>;
      if (isLinks) {
        payload = { content: serializeLinksContent(draftLinks) };
      } else if (isSpec) {
        payload = { content: draftContent };
      } else if (activePage) {
        payload = { page_id: activePage.id, content: draftContent };
      } else {
        payload = { generated_docs: draftContent };
      }
      const res = await fetch(
        `/api/products/${productId}/${entityPath(kind)}/${artifactId}`,
        {
          method: "PUT",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      if (res.status === 401) {
        router.replace(
          `/login?next=/products/${productId}/artifacts/${artifactId}`,
        );
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Save failed (${res.status})`);
      }
      const updated = (await res.json()) as Product;
      setProduct(updated);
      setEditing(false);
      setDirty(false);
      notify({ tone: "success", title: t.savedTitle ?? "Saved", message: t.savedMessage ?? "Documentation updated and re-indexed." });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : "Save failed",
      });
    } finally {
      setSaving(false);
    }
  };

  const entityLabel = isCodebase
    ? (tArt?.codebase?.label ?? "Codebase")
    : isDatabase
      ? (tArt?.database?.label ?? "Database")
      : isSpec
        ? (tArt?.spec?.label ?? "Spec")
        : (tArt?.links?.label ?? "Links");
  const entityTone =
    isCodebase || isDatabase ? "blue" : isSpec ? "green" : "yellow";
  const entityIcon =
    isCodebase ? <GitBranch size={18} weight="regular" /> :
    isDatabase ? <DatabaseIcon size={18} weight="regular" /> :
    isSpec ? <FileText size={18} weight="regular" /> :
    <LinkSimple size={18} weight="regular" />;

  const remove = async () => {
    if (!kind || !product || deleting) return;
    if (!confirm(t.deleteConfirm ?? "Delete this item?")) return;
    setDeleting(true);
    try {
      const res = await fetch(
        `/api/products/${productId}/${entityPath(kind)}/${artifactId}`,
        { method: "DELETE", credentials: "include" },
      );
      if (res.status === 401) {
        router.replace(
          `/login?next=/products/${productId}/artifacts/${artifactId}`,
        );
        return;
      }
      if (!res.ok) throw new Error(`Delete failed (${res.status})`);
      router.push(`/products/${productId}`);
    } catch (e) {
      notify({
        tone: "error",
        title: t.deleteFailedTitle ?? "Delete failed",
        message: e instanceof Error ? e.message : "Delete failed",
      });
    } finally {
      setDeleting(false);
    }
  };

  const empty = (isCodebase || isDatabase) ? !hasDocs : !hasRawContent;

  // Per-page regeneration: the forced page rides through the normal docgen
  // job (diff-reuse for the rest) and lands as a NEW doc version.
  const handleRegeneratePage = async () => {
    if (!kind || !activePage || regenJob || regenStarting) return;
    setConfirmRegen(false);
    setRegenStarting(true);
    const base = `/api/products/${productId}/${entityPath(kind)}/${artifactId}`;
    try {
      const res = await fetch(
        `${base}/pages/${encodeURIComponent(activePage.id)}/regenerate`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Generation failed (${res.status})`);
      }
      const jobId = data.job_id ?? null;
      if (!jobId) {
        await fetchProduct();
        return;
      }
      notify({
        tone: "info",
        title: t.regeneratePage ?? "Regenerate page",
        message: t.pageRegenStarted ?? "Page regeneration started…",
      });
      setRegenJob(jobId);
      const maxWaitMs = 30 * 60 * 1000;
      const startedAt = Date.now();
      while (Date.now() - startedAt < maxWaitMs) {
        await new Promise((r) => setTimeout(r, 2000));
        if (viewerAbortRef.current) return;
        const stRes = await fetch(
          `${base}/generate/status?job_id=${encodeURIComponent(jobId)}`,
          { credentials: "include", cache: "no-store" },
        );
        if (!stRes.ok) throw new Error(`Status failed (${stRes.status})`);
        const st = await stRes.json().catch(() => ({}));
        if (st.status === "succeeded") {
          notify({
            tone: "success",
            title: t.regeneratePage ?? "Regenerate page",
            message: st.indexing_message || (t.pageRegenDone ?? "Page regenerated."),
          });
          await fetchProduct();
          await fetchVersions();
          return;
        }
        if (st.status === "failed") {
          throw new Error(st.error || st.indexing_message || (t.pageRegenFailed ?? "Page regeneration failed."));
        }
        if (st.status === "cancelled") {
          notify({
            tone: "info",
            title: t.regeneratePage ?? "Regenerate page",
            message: st.indexing_message || (t.pageRegenCancelled ?? "Regeneration cancelled."),
          });
          await fetchProduct();
          return;
        }
      }
      throw new Error(t.pageRegenFailed ?? "Page regeneration timed out.");
    } catch (e) {
      if (!viewerAbortRef.current) {
        notify({
          tone: "error",
          title: t.regeneratePage ?? "Regenerate page",
          message: e instanceof Error ? e.message : (t.pageRegenFailed ?? "Page regeneration failed."),
        });
      }
    } finally {
      if (!viewerAbortRef.current) {
        setRegenJob(null);
        setRegenStarting(false);
      }
    }
  };

  // Cooperative cancel of the per-page regen job (the worker restores the
  // current version — no new version is appended).
  const handleCancelRegen = async () => {
    if (!kind || !regenJob) return;
    try {
      const res = await fetch(
        `/api/products/${productId}/${entityPath(kind)}/${artifactId}/generate/cancel`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        },
      );
      if (!res.ok) throw new Error(`Failed to cancel (${res.status})`);
    } catch (e) {
      notify({
        tone: "error",
        title: t.regeneratePage ?? "Regenerate page",
        message: e instanceof Error ? e.message : "Failed to cancel.",
      });
    }
  };

  // Rollback: writes the archived snapshot back and appends a new rollback
  // version; the returned Product replaces local state wholesale.
  const handleRestore = async (version: number) => {
    if (!kind || restoring) return;
    setConfirmRestore(null);
    setRestoring(true);
    try {
      const res = await fetch(
        `/api/products/${productId}/${entityPath(kind)}/${artifactId}/versions/${version}/restore`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Rollback failed (${res.status})`);
      }
      setProduct(data as Product);
      setViewVersion(null);
      notify({
        tone: "success",
        title: t.versionsTitle ?? "Versions",
        message: t.rollbackDone ?? "Rolled back.",
      });
      await fetchVersions();
    } catch (e) {
      notify({
        tone: "error",
        title: t.versionsTitle ?? "Versions",
        message: e instanceof Error ? e.message : (t.rollbackFailed ?? "Rollback failed."),
      });
    } finally {
      setRestoring(false);
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <AppHeader />

      <main className="mx-auto px-6 py-12">
        <Reveal>
          <Link
            href={`/products/${productId}`}
            className="inline-flex items-center gap-1 text-xs font-medium text-muted transition-colors hover:text-ink"
          >
            <ArrowLeft size={14} weight="bold" />
            {t.backToProduct ?? "Back to product"}
          </Link>
        </Reveal>

        {isLoading ? (
          <div className="mt-12 flex items-center gap-2 text-sm text-muted">
            <Spinner /> {t.loading ?? "Loading…"}
          </div>
        ) : error || !entity || !kind ? (
          <div className="mt-12">
            <EmptyState
              icon={<FileText size={20} weight="regular" />}
              title={t.notAvailable ?? "Not available"}
              description={error || (t.notFoundFallback ?? "This item could not be found.")}
              action={
                <Link href={`/products/${productId}`}>
                  <Button>{t.backToProduct ?? "Back to product"}</Button>
                </Link>
              }
            />
          </div>
        ) : (
          <>
            {/* Header + Verified */}
            <Reveal className="mt-6">
              <div className="flex flex-col gap-4 border-b border-divider pb-6 md:flex-row md:items-center md:justify-between">
                <div className="flex items-center gap-3">
                  <span className="flex h-10 w-10 items-center justify-center rounded-md bg-surface-2 text-ink">
                    {entityIcon}
                  </span>
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Tag tone={entityTone}>{entityLabel}</Tag>
                      {isSpec && spec?.kind && (
                        <Tag tone="neutral">{spec.kind}</Tag>
                      )}
                      <span className="font-mono text-xs text-muted">
                        {entity.id}
                      </span>
                      {verified && (
                        <VerifiedBadge verified={verified} verifiedBy={verifiedBy} />
                      )}
                    </div>
                    <h1 className="mt-2 font-editorial text-2xl tracking-tight text-ink">
                      {entity.name}
                    </h1>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {(isCodebase || isDatabase) && versions && versions.versions.length > 0 && (
                    <Select
                      value={viewVersion === null ? "" : String(viewVersion)}
                      onChange={(e) =>
                        setViewVersion(e.target.value ? Number(e.target.value) : null)
                      }
                      disabled={editing || regenStarting || regenJob !== null}
                      className="w-56 text-sm"
                    >
                      <option value="">
                        {fmt(t.currentVersion ?? "Current (v{n})", {
                          n: String(versions.current_version ?? "—"),
                        })}
                      </option>
                      {versions.versions.map((v) => (
                        <option key={v.version} value={v.version}>
                          {`v${v.version} · ${v.source}${v.created_at ? ` · ${new Date(v.created_at).toLocaleString()}` : ""}`}
                        </option>
                      ))}
                    </Select>
                  )}
                  {viewingArchive && (
                    <>
                      <Tag tone="yellow">
                        {fmt(t.archivedVersion ?? "Version v{n} (archived)", { n: String(viewVersion) })}
                      </Tag>
                      <Button
                        type="button"
                        variant="subtle"
                        size="sm"
                        onClick={() => setConfirmRestore(viewVersion)}
                        disabled={restoring}
                      >
                        <ArrowCounterClockwise size={14} weight="regular" />
                        {t.rollback ?? "Rollback"}
                      </Button>
                    </>
                  )}
                  {canEdit && !editing && (
                    <Button type="button" variant="subtle" size="sm" onClick={startEditing}>
                      <PencilSimple size={14} weight="regular" />
                      {t.edit ?? "Edit"}
                    </Button>
                  )}
                  {editing && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setEditing(false);
                        setDirty(false);
                      }}
                      disabled={saving}
                    >
                      {tc.cancel ?? "Cancel"}
                    </Button>
                  )}
                  {/* Page-bearing kinds verify per page instead (action row
                      below); the artifact-level button stays for spec/links. */}
                  {(isSpec || isLinks) && (
                    <VerifiedButton
                      verified={verified}
                      verifyUrl={`/api/products/${productId}/${entityPath(kind)}/${artifactId}/verify`}
                      ownerId={product?.owner_id ?? null}
                      onVerified={(next) => {
                        setVerified(next.verified);
                        setVerifiedBy(next.verified_by ?? null);
                      }}
                    />
                  )}
                  <IconButton
                    type="button"
                    aria-label={t.delete ?? "Delete"}
                    title={t.delete ?? "Delete"}
                    onClick={remove}
                    disabled={deleting}
                  >
                    {deleting ? <Spinner /> : <Trash size={14} weight="regular" />}
                  </IconButton>
                </div>
              </div>
            </Reveal>

            {/* Empty state */}
            {empty && (
              <div className="mt-10">
                <EmptyState
                  icon={<FileText size={20} weight="regular" />}
                  title={t.noDocsTitle ?? "No content yet"}
                  description={
                    isCodebase
                      ? (t.noDocsDesc ?? "")
                      : isDatabase
                        ? (t.noDbDocsDesc ?? "")
                        : (t.noRawContentDesc ?? "Add content from the edit button above.")
                  }
                  action={
                    isCodebase || isDatabase ? (
                      <Link href={`/products/${productId}`}>
                        <Button>{t.generateOnProductPage ?? "Generate on product page"}</Button>
                      </Link>
                    ) : (
                      <Button variant="subtle" size="sm" onClick={startEditing}>
                        <PencilSimple size={14} weight="regular" />
                        {t.edit ?? "Edit"}
                      </Button>
                    )
                  }
                />
              </div>
            )}

            {!empty && isSpec && (
              <div className="mt-8 flex flex-col gap-8">
                <Card className="p-6 md:p-10">
                  {editing ? (
                    <div className="flex flex-col gap-4">
                      <h2 className="font-editorial text-2xl tracking-tight text-ink">
                        {t.specEditTitle ?? "Edit specification"}
                      </h2>
                      <Textarea
                        value={draftContent}
                        onChange={(e) => {
                          setDraftContent(e.target.value);
                          setDirty(true);
                        }}
                        rows={18}
                        className="font-mono text-xs"
                        placeholder={t.specPlaceholder ?? ""}
                      />
                      <EditorSaveBar
                        saving={saving}
                        dirty={dirty}
                        onSave={save}
                        saveLabel={t.saveDocument ?? "Save"}
                      />
                    </div>
                  ) : (
                    <SpecViewer content={spec?.content || ""} kind={spec?.kind} />
                  )}
                </Card>
              </div>
            )}

            {!empty && isLinks && (
              <div className="mt-8 flex flex-col gap-8">
                <Card className="p-6 md:p-10">
                  {editing ? (
                    <div className="flex flex-col gap-4">
                      <div className="flex items-center justify-between gap-2">
                        <h2 className="font-editorial text-2xl tracking-tight text-ink">
                          {t.linksEditTitle ?? "Edit links"}
                        </h2>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => setDraftLinks((rows) => [...rows, { url: "", description: "" }])}
                        >
                          <Plus size={14} weight="bold" />
                          {t.linksAddRow ?? "Add link"}
                        </Button>
                      </div>
                      <div className="grid gap-2">
                        {draftLinks.map((row, idx) => (
                          <div
                            key={idx}
                            className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_1fr_auto]"
                          >
                            <Input
                              value={row.url}
                              onChange={(e) =>
                                setDraftLinks((rows) =>
                                  rows.map((r, i) => (i === idx ? { ...r, url: e.target.value } : r)),
                                )
                              }
                              placeholder={t.linksUrlPlaceholder ?? "https://…"}
                            />
                            <Input
                              value={row.description ?? ""}
                              onChange={(e) =>
                                setDraftLinks((rows) =>
                                  rows.map((r, i) => (i === idx ? { ...r, description: e.target.value } : r)),
                                )
                              }
                              placeholder={t.linksDescPlaceholder ?? "Description"}
                            />
                            <IconButton
                              type="button"
                              aria-label={t.linksRemoveRow ?? "Remove link"}
                              title={t.linksRemoveRow ?? "Remove link"}
                              disabled={draftLinks.length <= 1}
                              onClick={() =>
                                setDraftLinks((rows) =>
                                  rows.length <= 1
                                    ? [{ url: "", description: "" }]
                                    : rows.filter((_, i) => i !== idx),
                                )
                              }
                            >
                              <Trash size={16} weight="regular" />
                            </IconButton>
                          </div>
                        ))}
                      </div>
                      <EditorSaveBar
                        saving={saving}
                        dirty={dirty}
                        onSave={save}
                        saveLabel={t.saveDocument ?? "Save"}
                      />
                    </div>
                  ) : (
                    <LinksViewer content={linksEntity?.content || ""} />
                  )}
                </Card>
              </div>
            )}

            {/* Databases render exactly like codebases: page nav + markdown
                (Mermaid inside) with the generated summary as fallback. */}
            {!empty && (isCodebase || isDatabase) && (
              <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[240px_1fr]">
                <aside className="lg:sticky lg:top-20 lg:self-start">
                  <SectionHeader title={t.pages ?? "Pages"} className="mb-3" />
                  {pages.length > 0 ? (
                    <nav className="flex flex-col gap-0.5">
                      {pageTree.roots.map((p) => {
                        const isActive = p.id === activePageId;
                        const children = pageTree.childrenByParent.get(p.id) ?? [];
                        const open = Boolean(openGroups[p.id]);
                        return (
                          <div key={p.id} className="flex flex-col gap-0.5">
                            <div className="flex items-center gap-1">
                              {children.length > 0 ? (
                                <button
                                  type="button"
                                  aria-expanded={open}
                                  aria-label={p.title}
                                  onClick={() =>
                                    setOpenGroups((g) => ({ ...g, [p.id]: !open }))
                                  }
                                  className="shrink-0 rounded p-1 text-muted transition-colors hover:bg-surface-2 hover:text-ink"
                                >
                                  <CaretDown
                                    size={12}
                                    weight="bold"
                                    className={cn(
                                      "transition-transform",
                                      !open && "-rotate-90",
                                    )}
                                  />
                                </button>
                              ) : (
                                <span className="w-5 shrink-0" aria-hidden />
                              )}
                              <button
                                onClick={() => setActivePageId(p.id)}
                                disabled={editing}
                                className={cn(
                                  "flex-1 rounded-md px-2 py-2 text-left text-sm transition-colors",
                                  "disabled:cursor-not-allowed disabled:opacity-50",
                                  isActive
                                    ? "bg-surface-2 font-medium text-ink"
                                    : "text-muted hover:bg-surface-2 hover:text-ink",
                                )}
                              >
                                {p.title}
                                {p.verified ? (
                                  <SealCheck
                                    size={11}
                                    weight="fill"
                                    className="ml-1 inline text-tag-green-fg"
                                  />
                                ) : null}
                                {children.length > 0 && (
                                  <span className="ml-1.5 font-mono text-[11px] text-muted">
                                    {children.length}
                                  </span>
                                )}
                              </button>
                            </div>
                            {children.length > 0 && open && (
                              <div
                                className="ml-3 flex flex-col gap-0.5 border-l border-divider pl-2"
                                role="group"
                                aria-label={p.title}
                              >
                                {children.map((c) => {
                                  const childActive = c.id === activePageId;
                                  return (
                                    <button
                                      key={c.id}
                                      onClick={() => setActivePageId(c.id)}
                                      disabled={editing}
                                      className={cn(
                                        "rounded-md px-3 py-1.5 text-left text-[13px] leading-snug transition-colors",
                                        "disabled:cursor-not-allowed disabled:opacity-50",
                                        childActive
                                          ? "bg-surface-2 font-medium text-ink"
                                          : "text-muted hover:bg-surface-2 hover:text-ink",
                                      )}
                                    >
                                      {c.title}
                                      {c.verified ? (
                                        <SealCheck
                                          size={11}
                                          weight="fill"
                                          className="ml-1 inline text-tag-green-fg"
                                        />
                                      ) : null}
                                    </button>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </nav>
                  ) : (
                    <p className="text-xs text-muted">
                      {t.noPagesFallback ?? "No structured pages — showing the generated summary."}
                    </p>
                  )}
                </aside>

                <div className="flex min-w-0 flex-col gap-8">
                  <div className="flex flex-col gap-4">
                    {(isCodebase || isDatabase) && !editing && (
                      <div className="flex items-center justify-end gap-2">
                        {regenStarting || regenJob ? (
                          <>
                            <span className="flex items-center gap-2 font-mono text-xs text-muted">
                              <Spinner className="h-3.5 w-3.5" />
                              {t.regeneratingPage ?? "Regenerating page…"}
                            </span>
                            {regenJob && (
                              <Button
                                type="button"
                                variant="subtle"
                                size="sm"
                                onClick={handleCancelRegen}
                              >
                                <StopCircle size={14} weight="fill" />
                                {t.stopGeneration ?? "Stop"}
                              </Button>
                            )}
                          </>
                        ) : (
                          activePage &&
                          !viewingArchive && (
                            <>
                              <VerifiedButton
                                verified={Boolean(activePage.verified)}
                                verifyUrl={`/api/products/${productId}/${entityPath(kind)}/${artifactId}/pages/${encodeURIComponent(activePage.id)}/verify`}
                                ownerId={product?.owner_id ?? null}
                                mapResponse={(data) => {
                                  const p = findPageInProduct(
                                    data as Product,
                                    artifactId,
                                    activePage.id,
                                  );
                                  return {
                                    verified: p?.verified ?? true,
                                    verified_by: p?.verified_by ?? null,
                                  };
                                }}
                                onResponse={(data) => setProduct(data as Product)}
                              />
                              <Button
                                type="button"
                                variant="subtle"
                                size="sm"
                                onClick={() => setConfirmRegen(true)}
                              >
                                <Lightning size={14} weight="fill" />
                                {t.regeneratePage ?? "Regenerate page"}
                              </Button>
                            </>
                          )
                        )}
                      </div>
                    )}
                  </div>
                  <Card className="p-6 md:p-10">
                    {editing ? (
                      <div className="flex flex-col gap-4">
                        <h2 className="font-editorial text-2xl tracking-tight text-ink">
                          {activePage ? activePage.title : (t.generatedDocs ?? "Generated documentation")}
                        </h2>
                        <MarkdownEditor
                          value={draftContent}
                          onChange={(v) => {
                            setDraftContent(v);
                            setDirty(true);
                          }}
                        />
                        <EditorSaveBar
                          saving={saving}
                          dirty={dirty}
                          onSave={save}
                          saveLabel={activePage ? (t.savePage ?? "Save page") : (t.saveDocument ?? "Save document")}
                        />
                      </div>
                    ) : activePage ? (
                      <article className="prose-editor max-w-none">
                        <div className="flex flex-wrap items-center gap-3">
                          <h2 className="font-editorial text-2xl tracking-tight text-ink">
                            {activePage.title}
                          </h2>
                          {activePage.verified ? (
                            <VerifiedBadge
                              verified={activePage.verified}
                              verifiedBy={activePage.verified_by ?? null}
                            />
                          ) : null}
                        </div>
                        <Markdown content={stripProvenanceBlock(activePage.content || "")} />
                      </article>
                    ) : (
                      <article className="prose-editor max-w-none">
                        <h2 className="font-editorial text-2xl tracking-tight text-ink">
                          {t.generatedDocs ?? "Generated documentation"}
                        </h2>
                        <Markdown
                          content={stripProvenanceBlock(
                            (viewingArchive ? versionDetail?.generated_docs : undefined) ||
                              codebase?.generated_docs ||
                              databaseEntity?.generated_docs ||
                              "",
                          )}
                        />
                      </article>
                    )}
                  </Card>
                  {!editing && !viewingArchive && activePage?.provenance ? (
                    <ProvenancePanel provenance={activePage.provenance} />
                  ) : null}
                </div>
              </div>
            )}
          </>
        )}

        {/* Per-page regeneration confirmation */}
        <Modal
          open={confirmRegen}
          onClose={() => setConfirmRegen(false)}
          title={t.confirmPageRegenTitle ?? "Regenerate page?"}
          footer={null}
        >
          <p className="text-sm text-muted">
            {t.confirmPageRegenText ??
              "A new documentation version will be generated for this page."}
          </p>
          <div className="mt-6 flex items-center justify-end gap-2">
            <Button variant="ghost" onClick={() => setConfirmRegen(false)}>
              {tc.cancel ?? "Cancel"}
            </Button>
            <Button onClick={() => void handleRegeneratePage()}>
              <Lightning size={16} weight="fill" />
              {t.regeneratePage ?? "Regenerate page"}
            </Button>
          </div>
        </Modal>

        {/* Rollback confirmation — restores the snapshot as a new version */}
        <Modal
          open={confirmRestore !== null}
          onClose={() => setConfirmRestore(null)}
          title={t.rollbackConfirmTitle ?? "Roll back to this version?"}
          footer={null}
        >
          <p className="text-sm text-muted">
            {t.rollbackConfirmText ??
              "The current documentation will be replaced by this version; the rollback itself is saved as a new version."}
          </p>
          <div className="mt-6 flex items-center justify-end gap-2">
            <Button variant="ghost" onClick={() => setConfirmRestore(null)}>
              {tc.cancel ?? "Cancel"}
            </Button>
            <Button
              onClick={() => {
                if (confirmRestore !== null) void handleRestore(confirmRestore);
              }}
              disabled={restoring}
            >
              {restoring ? <Spinner /> : <ArrowCounterClockwise size={16} weight="regular" />}
              {t.rollback ?? "Rollback"}
            </Button>
          </div>
        </Modal>
      </main>
    </div>
  );
}

const PROV_BLOCK_HEADING_RE = /^(#{2,4})\s*(.+?)\s*$/;

/**
 * Hides the legacy in-content "Провенанс и проверка" block at render time —
 * new pages carry it in provenance.report, rendered by the verification
 * panel under the text. Pure display concern: stored content is untouched.
 */
function stripProvenanceBlock(content: string): string {
  const lines = content.split("\n");
  let start = -1;
  let level = 0;
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = PROV_BLOCK_HEADING_RE.exec(lines[i]);
    if (!m || !m[2]) continue;
    const title = m[2].toLowerCase();
    const isProvenance = title.includes("провенанс") || title.includes("provenance");
    const isVerification = title.includes("проверка") || title.includes("verification");
    if (isProvenance && isVerification) {
      start = i;
      level = m[1]?.length ?? 3;
      break;
    }
  }
  if (start === -1) return content;
  let end = lines.length;
  for (let j = start + 1; j < lines.length; j++) {
    const m = /^(#{1,6})\s/.exec(lines[j]);
    if (m && (m[1]?.length ?? 0) <= level) {
      end = j;
      break;
    }
  }
  const head = lines.slice(0, start).join("\n").replace(/\s+$/, "");
  const tail = lines.slice(end).join("\n").replace(/^\s+/, "");
  const kept = [head, tail].filter(Boolean).join("\n\n");
  return kept || content;
}

/** Find one doc page inside a product payload by entity + page id. */
function findPageInProduct(
  product: Product,
  entityId: string,
  pageId: string,
): ArtifactPage | undefined {
  const found = findEntity(product, entityId);
  const pages =
    found && (found.kind === "codebase" || found.kind === "database")
      ? normalizePages((found.entity as Codebase | Database).pages)
      : [];
  return pages.find((p) => p.id === pageId);
}

/** Find a codebase/spec/links/database entity by id across the product's lists. */
function findEntity(
  product: Product,
  entityId: string,
): {
  entity: Codebase | Spec | Links | Database | undefined;
  kind: EntityKind | undefined;
} {
  const c = product.codebases.find((x) => x.id === entityId);
  if (c) return { entity: c, kind: "codebase" };
  const s = product.specs.find((x) => x.id === entityId);
  if (s) return { entity: s, kind: "spec" };
  const l = product.links.find((x) => x.id === entityId);
  if (l) return { entity: l, kind: "links" };
  const d = (product.databases ?? []).find((x) => x.id === entityId);
  if (d) return { entity: d, kind: "database" };
  return { entity: undefined, kind: undefined };
}
