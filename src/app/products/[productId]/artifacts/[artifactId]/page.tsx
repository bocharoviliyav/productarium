"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  Article,
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
  artifactTypeIcon,
  artifactTypeMeta,
  type Artifact,
  type ArtifactPage,
  type LinkItem,
  type Product,
  normalizePages,
  parseLinksContent,
  serializeLinksContent,
} from "@/lib/types";

const Markdown = dynamic(() => import("@/components/Markdown"), {
  ssr: false,
  loading: () => <div className="text-sm text-muted">{"Loading…"}</div>,
});

/** Map a Phosphor icon name to a rendered node. */
const ICON_BY_NAME: Record<string, React.ReactNode> = {
  GitBranch: <GitBranch size={18} weight="regular" />,
  FileText: <FileText size={18} weight="regular" />,
  LinkSimple: <LinkSimple size={18} weight="regular" />,
  Article: <Article size={18} weight="regular" />,
};

/** Render the icon for an arbitrary artifact type (incl. legacy ones). */
function artifactIconFor(type: string): React.ReactNode {
  return ICON_BY_NAME[artifactTypeIcon(type)] ?? <FileText size={18} weight="regular" />;
}

export default function ArtifactDocsViewer() {
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

  // WYSIWYG editor state. `editing` swaps the read-only render for the editor;
  // `draftContent` is seeded from the active page's content (or generated_docs
  // when there are no structured pages, or raw `content` for spec). Saving PUTs
  // the doc back to the backend, which re-indexes it into cognee.
  const [editing, setEditing] = useState(false);
  const [draftContent, setDraftContent] = useState("");
  const [draftLinks, setDraftLinks] = useState<LinkItem[]>([]);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

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
      if (!res.ok) throw new Error(`Failed to load artifact (${res.status})`);
      const data = (await res.json()) as Product;
      setProduct(data);
      const art = data.artifacts.find((a) => a.id === artifactId);
      setVerified(Boolean(art?.verified));
      setVerifiedBy(art?.verified_by ?? null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to load artifact.";
      setError(msg);
      notify({ tone: "error", title: t.loadFailedTitle ?? "Failed to load artifact", message: msg });
    } finally {
      setIsLoading(false);
    }
  }, [productId, artifactId, router, notify]);

  useEffect(() => {
    fetchProduct();
  }, [fetchProduct]);

  const artifact: Artifact | undefined = useMemo(
    () => product?.artifacts.find((a) => a.id === artifactId),
    [product, artifactId],
  );

  const pages: ArtifactPage[] = useMemo(
    () => (artifact ? normalizePages(artifact.pages) : []),
    [artifact],
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

  const meta = artifact ? artifactTypeMeta(artifact.type) : null;
  const artType = artifact?.type;
  const isCodebase = artType === "codebase";
  const isSpec = artType === "spec";
  const isLinks = artType === "links";
  // codebase: generated docs/pages are editable. spec/links: the raw `content`
  // is editable (no generation step). Legacy types fall back to read-only.
  const hasDocs = Boolean(artifact?.generated_docs || pages.length > 0);
  const hasRawContent = Boolean(artifact?.content);
  const canEdit =
    (isCodebase && hasDocs) || ((isSpec || isLinks) && hasRawContent);

  // Seed the editor draft whenever the source changes, unless the user is
  // mid-edit (don't clobber unsaved changes).
  useEffect(() => {
    if (editing) return;
    if (isLinks) {
      setDraftLinks(
        parseLinksContent(artifact?.content).length
          ? parseLinksContent(artifact?.content)
          : [{ url: "", description: "" }],
      );
    } else {
      setDraftContent(
        activePage ? activePage.content || "" : artifact?.content || artifact?.generated_docs || "",
      );
    }
    setDirty(false);
  }, [activePage, artifact, editing, isLinks]);

  const startEditing = () => {
    if (isLinks) {
      setDraftLinks(
        parseLinksContent(artifact?.content).length
          ? parseLinksContent(artifact?.content)
          : [{ url: "", description: "" }],
      );
    } else {
      setDraftContent(
        activePage ? activePage.content || "" : artifact?.content || artifact?.generated_docs || "",
      );
    }
    setDirty(false);
    setEditing(true);
  };

  const save = async () => {
    if (!artifact || saving) return;
    setSaving(true);
    try {
      let payload: Record<string, unknown>;
      if (isLinks) {
        // Persist the repeater rows as the artifact's raw content (JSON).
        payload = { raw_content: serializeLinksContent(draftLinks) };
      } else if (isSpec) {
        // Spec edits update the raw spec source.
        payload = { raw_content: draftContent };
      } else if (activePage) {
        payload = { page_id: activePage.id, content: draftContent };
      } else {
        payload = { generated_docs: draftContent };
      }
      const res = await fetch(
        `/api/products/${productId}/artifacts/${artifactId}`,
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
            <Spinner /> {t.loading ?? "Loading artifact…"}
          </div>
        ) : error || !artifact || !meta ? (
          <div className="mt-12">
            <EmptyState
              icon={<FileText size={20} weight="regular" />}
              title={t.notAvailable ?? "Artifact not available"}
              description={error || (t.notFoundFallback ?? "This artifact could not be found.")}
              action={
                <Link href={`/products/${productId}`}>
                  <Button>{t.backToProduct ?? "Back to product"}</Button>
                </Link>
              }
            />
          </div>
        ) : (
          <>
            {/* Header + Verified (item 5) */}
            <Reveal className="mt-6">
              <div className="flex flex-col gap-4 border-b border-divider pb-6 md:flex-row md:items-center md:justify-between">
                <div className="flex items-center gap-3">
                  <span className="flex h-10 w-10 items-center justify-center rounded-md bg-surface-2 text-ink">
                    {artifactIconFor(artifact.type)}
                  </span>
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Tag tone={meta.tone}>{(tArt?.[artifact.type]?.label as string) ?? meta.label}</Tag>
                      {artifact.kind && (
                        <Tag tone="neutral">{artifact.kind}</Tag>
                      )}
                      <span className="font-mono text-xs text-muted">
                        {artifact.id}
                      </span>
                      {verified && (
                        <VerifiedBadge
                          verified={verified}
                          verifiedBy={verifiedBy}
                        />
                      )}
                    </div>
                    <h1 className="mt-2 font-editorial text-2xl tracking-tight text-ink">
                      {artifact.name}
                    </h1>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {canEdit && !editing && (
                    <Button
                      type="button"
                      variant="subtle"
                      size="sm"
                      onClick={startEditing}
                    >
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
                    verifyUrl={`/api/products/${productId}/artifacts/${artifactId}/verify`}
                    ownerId={product?.owner_id ?? null}
                    onVerified={(next) => {
                      setVerified(next.verified);
                      setVerifiedBy(next.verified_by ?? null);
                    }}
                  />
                </div>
              </div>
            </Reveal>

            {/* Empty state: nothing to show yet. Codebase needs generation;
             spec/links need authored content; legacy types need generated_docs. */}
            {(() => {
              const empty = isCodebase ? !hasDocs : isSpec || isLinks ? !hasRawContent : !hasDocs;
              if (empty) {
                return (
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
                );
              }
              return null;
            })()}

            {(() => {
              const empty = isCodebase ? !hasDocs : isSpec || isLinks ? !hasRawContent : !hasDocs;
              if (empty) return null;

              // Spec: single-column structured viewer / raw editor.
              if (isSpec) {
                return (
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
                        <SpecViewer
                          content={artifact.content || ""}
                          kind={artifact.kind ?? undefined}
                        />
                      )}
                    </Card>
                    <ExpertCta
                      t={t}
                      productId={productId}
                    />
                  </div>
                );
              }

              // Links: single-column table / repeater editor.
              if (isLinks) {
                return (
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
                              onClick={() =>
                                setDraftLinks((rows) => [
                                  ...rows,
                                  { url: "", description: "" },
                                ])
                              }
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
                                      rows.map((r, i) =>
                                        i === idx ? { ...r, url: e.target.value } : r,
                                      ),
                                    )
                                  }
                                  placeholder={t.linksUrlPlaceholder ?? "https://…"}
                                />
                                <Input
                                  value={row.description ?? ""}
                                  onChange={(e) =>
                                    setDraftLinks((rows) =>
                                      rows.map((r, i) =>
                                        i === idx
                                          ? { ...r, description: e.target.value }
                                          : r,
                                      ),
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
                        <LinksViewer content={artifact.content || ""} />
                      )}
                    </Card>
                    <ExpertCta t={t} productId={productId} />
                  </div>
                );
              }

              // Codebase (and legacy generated-doc types): page nav + markdown.
              return (
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
                          <Markdown content={artifact.generated_docs || ""} />
                        </article>
                      )}
                    </Card>
                    <ExpertCta t={t} productId={productId} />
                  </div>
                </div>
              );
            })()}
          </>
        )}
      </main>
    </div>
  );
}

/** Compact "Ask expert" CTA card reused across artifact viewer layouts. */
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
