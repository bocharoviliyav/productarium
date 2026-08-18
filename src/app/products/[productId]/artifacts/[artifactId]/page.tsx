"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  Chats,
  FileText,
  GitBranch,
  LinkSimple,
  PencilSimple,
  Plus,
  Trash,
} from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { VerifiedBadge, VerifiedButton } from "@/components/VerifiedBadge";
import { MarkdownEditor, EditorSaveBar } from "@/components/MarkdownEditor";
import { SpecViewer } from "@/components/SpecViewer";
import { LinksViewer } from "@/components/LinksViewer";
import dynamic from "next/dynamic";
import {
  Button,
  Card,
  EmptyState,
  IconButton,
  Input,
  Reveal,
  SectionHeader,
  Spinner,
  Tag,
  Textarea,
  cn,
} from "@/components/ui";
import {
  type ArtifactPage,
  type Codebase,
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
  const { messages } = useLanguage();
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

  const pages: ArtifactPage[] = useMemo(
    () => (entity && kind === "codebase" ? normalizePages((entity as Codebase).pages) : []),
    [entity, kind],
  );

  useEffect(() => {
    if (pages.length > 0 && !activePageId) {
      setActivePageId(pages[0].id);
    }
  }, [pages, activePageId]);

  const activePage = useMemo(
    () => pages.find((p) => p.id === activePageId) ?? null,
    [pages, activePageId],
  );

  const isCodebase = kind === "codebase";
  const isSpec = kind === "spec";
  const isLinks = kind === "links";
  const codebase = entity as Codebase | undefined;
  const spec = entity as Spec | undefined;
  const linksEntity = entity as Links | undefined;

  const hasDocs = Boolean(codebase?.generated_docs || pages.length > 0);
  const hasRawContent = Boolean(spec?.content || linksEntity?.content);
  const canEdit =
    (isCodebase && hasDocs) || ((isSpec || isLinks) && hasRawContent);

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
          : spec?.content || codebase?.generated_docs || "",
      );
    }
    setDirty(false);
  }, [activePage, codebase, spec, linksEntity, editing, isLinks]);

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
          : spec?.content || codebase?.generated_docs || "",
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
    : isSpec
      ? (tArt?.spec?.label ?? "Spec")
      : (tArt?.links?.label ?? "Links");
  const entityTone = isCodebase ? "blue" : isSpec ? "green" : "yellow";
  const entityIcon =
    isCodebase ? <GitBranch size={18} weight="regular" /> :
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

  const empty = isCodebase ? !hasDocs : !hasRawContent;

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
                <div className="flex items-center gap-2">
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
                  <VerifiedButton
                    verified={verified}
                    verifyUrl={`/api/products/${productId}/${entityPath(kind)}/${artifactId}/verify`}
                    ownerId={product?.owner_id ?? null}
                    onVerified={(next) => {
                      setVerified(next.verified);
                      setVerifiedBy(next.verified_by ?? null);
                    }}
                  />
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
                      : (t.noRawContentDesc ?? "Add content from the edit button above.")
                  }
                  action={
                    isCodebase ? (
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
                <ExpertCta t={t} productId={productId} />
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
                <ExpertCta t={t} productId={productId} />
              </div>
            )}

            {!empty && isCodebase && (
              <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[240px_1fr]">
                <aside className="lg:sticky lg:top-20 lg:self-start">
                  <SectionHeader title={t.pages ?? "Pages"} className="mb-3" />
                  {pages.length > 0 ? (
                    <nav className="flex flex-col gap-0.5">
                      {pages.map((p) => {
                        const isActive = p.id === activePageId;
                        return (
                          <button
                            key={p.id}
                            onClick={() => setActivePageId(p.id)}
                            disabled={editing}
                            className={cn(
                              "rounded-md px-3 py-2 text-left text-sm transition-colors",
                              "disabled:cursor-not-allowed disabled:opacity-50",
                              isActive
                                ? "bg-surface-2 font-medium text-ink"
                                : "text-muted hover:bg-surface-2 hover:text-ink",
                            )}
                          >
                            {p.title}
                          </button>
                        );
                      })}
                    </nav>
                  ) : (
                    <p className="text-xs text-muted">
                      {t.noPagesFallback ?? "No structured pages — showing the generated summary."}
                    </p>
                  )}
                </aside>

                <div className="flex flex-col gap-8">
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
                        <h2 className="font-editorial text-2xl tracking-tight text-ink">
                          {activePage.title}
                        </h2>
                        <Markdown content={activePage.content || ""} />
                      </article>
                    ) : (
                      <article className="prose-editor max-w-none">
                        <h2 className="font-editorial text-2xl tracking-tight text-ink">
                          {t.generatedDocs ?? "Generated documentation"}
                        </h2>
                        <Markdown content={codebase?.generated_docs || ""} />
                      </article>
                    )}
                  </Card>
                  <ExpertCta t={t} productId={productId} />
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

/** Find a codebase/spec/links entity by id across the product's three lists. */
function findEntity(
  product: Product,
  entityId: string,
): { entity: Codebase | Spec | Links | undefined; kind: EntityKind | undefined } {
  const c = product.codebases.find((x) => x.id === entityId);
  if (c) return { entity: c, kind: "codebase" };
  const s = product.specs.find((x) => x.id === entityId);
  if (s) return { entity: s, kind: "spec" };
  const l = product.links.find((x) => x.id === entityId);
  if (l) return { entity: l, kind: "links" };
  return { entity: undefined, kind: undefined };
}

/** Compact "Ask expert" CTA card reused across viewer layouts. */
function ExpertCta({
  t,
  productId,
}: {
  t: Record<string, string> | undefined;
  productId: string;
}) {
  return (
    <Card className="p-6 md:p-8">
      <SectionHeader
        title={t?.askExpertTitle ?? "Ask expert"}
        subtitle={t?.askExpertSubtitle ?? ""}
        action={
          <span className="inline-flex items-center gap-1.5 text-xs text-muted">
            <Chats size={14} weight="regular" />
            {t?.expertBadge ?? "expert"}
          </span>
        }
      />
      <div className="mt-4">
        <Link href={`/products/${productId}`}>
          <Button variant="subtle">
            {t?.openExpertChat ?? "Open expert chat"}
            <ArrowLeft size={14} weight="bold" className="rotate-180" />
          </Button>
        </Link>
      </div>
    </Card>
  );
}
