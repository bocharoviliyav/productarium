"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  Article,
  FileText,
  GitBranch,
  Lightning,
  LinkSimple,
  Plus,
  Sparkle,
  Trash,
} from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { ExpertChat } from "@/components/ExpertChat";
import { SummaryBlock } from "@/components/SummaryBlock";
import { KnowledgeTree } from "@/components/knowledge/KnowledgeTree";
import { useLanguage } from "@/contexts/LanguageContext";
import {
  Button,
  Card,
  EmptyState,
  IconButton,
  Input,
  Label,
  Reveal,
  SectionHeader,
  Select,
  Spinner,
  Tag,
  Textarea,
  cn,
} from "@/components/ui";
import {
  artifactTypeIcon,
  artifactTypeMeta,
  type Artifact,
  type ArtifactKind,
  type ArtifactType,
  type KnowledgeNode,
  type LinkItem,
  type Product,
  generateId,
  serializeLinksContent,
} from "@/lib/types";
import { useNotifications } from "@/contexts/NotificationContext";

// Artifact types the UI offers when creating a new artifact.
// `documentation` and `guides` are no longer creatable here (authored as
// knowledge pages); legacy artifacts of those types still render via the
// fallback meta/icon resolvers in lib/types.ts.
const ARTIFACT_TYPES: ArtifactType[] = ["codebase", "spec", "links"];

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

export default function ProductDetailPage() {
  const router = useRouter();
  const params = useParams<{ productId: string }>();
  const productId = params.productId;

  const [product, setProduct] = useState<Product | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Add-artifact form
  const [showForm, setShowForm] = useState(false);
  const [artName, setArtName] = useState("");
  const [artType, setArtType] = useState<ArtifactType>("codebase");
  const [artKind, setArtKind] = useState<ArtifactKind>("openapi");
  const [artRepoUrl, setArtRepoUrl] = useState("");
  const [artRepoType, setArtRepoType] = useState("github");
  const [artToken, setArtToken] = useState("");
  const [artContent, setArtContent] = useState("");
  // `links` artifact: repeater rows of {url, description} serialized to JSON.
  const [artLinks, setArtLinks] = useState<LinkItem[]>([{ url: "", description: "" }]);
  const [isSavingArtifact, setIsSavingArtifact] = useState(false);

  const [generatingId, setGeneratingId] = useState<string | null>(null);
  const [deletingArtId, setDeletingArtId] = useState<string | null>(null);

  const { notify } = useNotifications();
  const { messages, fmt } = useLanguage();
  const t = messages?.product ?? {};
  const tc = messages?.common ?? {};
  const tArt = messages?.artifactTypes ?? {};

  // Tracks whether a product is already loaded so refetch failures surface as
  // a toast instead of wiping the page with the fatal EmptyState.
  const productRef = useRef<Product | null>(null);
  useEffect(() => {
    productRef.current = product;
  }, [product]);

  // Abort flag for the generate-status polling loop so it stops cleanly when
  // the component unmounts (e.g. user navigates away mid-generation).
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
      const res = await fetch(`/api/products/${productId}`, {
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
      // First load (no product yet) is fatal -> EmptyState; later refetch
      // failures degrade to a toast so the existing page stays usable.
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

  const resetForm = () => {
    setArtName("");
    setArtType("codebase");
    setArtKind("openapi");
    setArtRepoUrl("");
    setArtRepoType("github");
    setArtToken("");
    setArtContent("");
    setArtLinks([{ url: "", description: "" }]);
  };

  const handleAddArtifact = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!product || !artName.trim() || isSavingArtifact) return;
    setIsSavingArtifact(true);
    try {
      const draft: Artifact = {
        id: generateId("art"),
        name: artName.trim(),
        type: artType,
        kind: artType === "spec" ? artKind : null,
        repo_url: artType === "codebase" ? artRepoUrl.trim() || null : null,
        repo_type: artType === "codebase" ? artRepoType : null,
        token: artType === "codebase" && artToken ? artToken : null,
        content:
          artType === "spec"
            ? artContent || null
            : artType === "links"
              ? serializeLinksContent(artLinks) || null
              : null,
        source: "manual",
      };
      const res = await fetch(`/api/products/${product.id}/artifacts`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Failed to add artifact (${res.status})`);
      }
      const updated = (await res.json()) as Product;
      setProduct(updated);
      resetForm();
      setShowForm(false);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to add artifact.";
      notify({ tone: "error", title: t.addArtifactFailedTitle ?? "Add artifact", message: msg });
    } finally {
      setIsSavingArtifact(false);
    }
  };

  const handleDeleteArtifact = async (artifactId: string) => {
    if (!product) return;
    if (!confirm(t.deleteArtifactConfirm ?? "Delete this artifact?")) return;
    setDeletingArtId(artifactId);
    try {
      const res = await fetch(
        `/api/products/${product.id}/artifacts/${artifactId}`,
        { method: "DELETE", credentials: "include" },
      );
      if (!res.ok) throw new Error(`Failed to delete (${res.status})`);
      const updated = (await res.json()) as Product;
      setProduct(updated);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to delete artifact.";
      notify({ tone: "error", title: t.deleteArtifactFailedTitle ?? "Delete artifact", message: msg });
    } finally {
      setDeletingArtId(null);
    }
  };

  const handleGenerate = async (artifactId: string) => {
    if (!product) return;
    setGeneratingId(artifactId);
    setError(null);
    try {
      const res = await fetch(
        `/api/products/${product.id}/artifacts/${artifactId}/generate`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ language: "en" }),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Generation failed (${res.status})`);
      }
      // Backward-compatible fast path: a synchronous 200 with generated_docs
      // (e.g. an alternate implementation) is handled immediately.
      if (data.generated_docs) {
        notify({ tone: "success", title: t.genTitle ?? "Generation", message: t.genDone ?? "Documentation generated." });
        await fetchProduct();
        return;
      }
      const jobId = data.job_id;
      if (!jobId) {
        notify({ tone: "info", title: t.genTitle ?? "Generation", message: data.message || data.status || (t.genTriggered ?? "Generation triggered.") });
        await fetchProduct();
        return;
      }
      // Async 202 + poll: the backend offloads heavy work (git clone, file
      // read, RLM) to a worker thread, so we poll the status endpoint until
      // the job succeeds or fails. Display is decoupled from the cognee
      // knowledge graph — once the job is "succeeded" the docs are committed
      // and shown immediately, even while cognify continues in the background.
      notify({ tone: "info", title: t.genTitle ?? "Generation", message: t.genStarted ?? "Generation started…" });
      const maxWaitMs = 30 * 60 * 1000;
      const startedAt = Date.now();
      while (Date.now() - startedAt < maxWaitMs) {
        if (generateAbortRef.current) return;
        await new Promise((r) => setTimeout(r, 2000));
        if (generateAbortRef.current) return;
        const stRes = await fetch(
          `/api/products/${product.id}/artifacts/${artifactId}/generate/status?job_id=${encodeURIComponent(jobId)}`,
          { credentials: "include", cache: "no-store" },
        );
        if (stRes.status === 404) {
          throw new Error(t.genJobNotFound ?? "Generation job not found.");
        }
        if (!stRes.ok) {
          throw new Error(fmt(t.genStatusFailed, { status: String(stRes.status) }));
        }
        const st = await stRes.json().catch(() => ({}));
        if (st.status === "succeeded") {
          // Docs are already committed and shown now; cognee indexing (if any)
          // continues in the background and never gates display. Indexing
          // status is surfaced only as an informational note, never an error.
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
        // queued / running -> keep polling.
      }
      throw new Error(t.genTimeout ?? "Generation timed out.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : (t.genFailed ?? "Generation failed.");
      notify({ tone: "error", title: t.genTitle ?? "Generation", message: msg });
    } finally {
      setGeneratingId(null);
    }
  };

  const onTreeSelect = (node: KnowledgeNode) => {
    if (node.node_type === "page") {
      router.push(`/products/${productId}/knowledge/${node.id}`);
    }
  };

  const artifactTypeOptions = useMemo(
    () =>
      ARTIFACT_TYPES.map((tp) => ({
        value: tp,
        label: (tArt?.[tp]?.label as string) ?? artifactTypeMeta(tp).label,
      })),
    [tArt],
  );

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <AppHeader />

      <main className="mx-auto px-6 py-16">
        {/* Back */}
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
                <Button
                  onClick={() => setShowForm((v) => !v)}
                  variant={showForm ? "ghost" : "primary"}
                >
                  <Plus size={16} weight="bold" />
                  {showForm ? tc.close ?? "Close" : t.addArtifact ?? "Add artifact"}
                </Button>
              </div>
            </Reveal>

            {/* Summary block (item 4) */}
            <Reveal className="mt-8">
              <SummaryBlock product={product} onRefresh={fetchProduct} />
            </Reveal>

            {/* Add artifact form */}
            {showForm && (
              <Reveal className="mt-8">
                <Card className="p-6">
                  <form onSubmit={handleAddArtifact} className="grid gap-5">
                    <div className="grid gap-5 md:grid-cols-[1fr_220px]">
                      <div>
                        <Label>{t.artifactName ?? "Artifact name"}</Label>
                        <Input
                          value={artName}
                          onChange={(e) => setArtName(e.target.value)}
                          placeholder={t.namePlaceholder ?? ""}
                          required
                          autoFocus
                        />
                      </div>
                      <div>
                        <Label>{t.type ?? "Type"}</Label>
                        <Select
                          value={artType}
                          onChange={(e) =>
                            setArtType(e.target.value as ArtifactType)
                          }
                        >
                          {artifactTypeOptions.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                        </Select>
                      </div>
                    </div>

                    {artType === "codebase" && (
                      <div className="grid gap-5 md:grid-cols-[1fr_180px]">
                        <div className="md:col-span-2">
                          <Label>{t.gitUrl ?? "Git URL"}</Label>
                          <Input
                            value={artRepoUrl}
                            onChange={(e) => setArtRepoUrl(e.target.value)}
                            placeholder={t.gitUrlPlaceholder ?? ""}
                          />
                        </div>
                        <div>
                          <Label>{t.provider ?? "Provider"}</Label>
                          <Select
                            value={artRepoType}
                            onChange={(e) => setArtRepoType(e.target.value)}
                          >
                            <option value="github">{t.github ?? "GitHub"}</option>
                            <option value="gitlab">{t.gitlab ?? "GitLab"}</option>
                          </Select>
                        </div>
                        <div>
                          <Label>{t.accessToken ?? "Access token (optional)"}</Label>
                          <Input
                            type="password"
                            value={artToken}
                            onChange={(e) => setArtToken(e.target.value)}
                            placeholder={t.tokenPlaceholder ?? ""}
                          />
                        </div>
                      </div>
                    )}

                    {artType === "spec" && (
                      <div className="grid gap-5">
                        <div>
                          <Label>{t.specKind ?? "Spec kind"}</Label>
                          <Select
                            value={artKind}
                            onChange={(e) =>
                              setArtKind(e.target.value as ArtifactKind)
                            }
                          >
                            <option value="openapi">{t.openapi ?? "OpenAPI (REST)"}</option>
                            <option value="asyncapi">{t.asyncapi ?? "AsyncAPI (events)"}</option>
                          </Select>
                        </div>
                        <div>
                          <Label>{t.specLabel ?? "Specification (JSON / YAML)"}</Label>
                          <Textarea
                            value={artContent}
                            onChange={(e) => setArtContent(e.target.value)}
                            placeholder={t.specPlaceholder ?? ""}
                            rows={10}
                          />
                        </div>
                      </div>
                    )}

                    {artType === "links" && (
                      <div className="grid gap-3">
                        <div className="flex items-center justify-between gap-2">
                          <Label className="mb-0">
                            {t.linksLabel ?? "Links"}
                          </Label>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() =>
                              setArtLinks((rows) => [
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
                          {artLinks.map((row, idx) => (
                            <div
                              key={idx}
                              className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_1fr_auto]"
                            >
                              <Input
                                value={row.url}
                                onChange={(e) =>
                                  setArtLinks((rows) =>
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
                                  setArtLinks((rows) =>
                                    rows.map((r, i) =>
                                      i === idx
                                        ? { ...r, description: e.target.value }
                                        : r,
                                    ),
                                  )
                                }
                                placeholder={
                                  t.linksDescPlaceholder ?? "Description"
                                }
                              />
                              <IconButton
                                type="button"
                                aria-label={t.linksRemoveRow ?? "Remove link"}
                                title={t.linksRemoveRow ?? "Remove link"}
                                disabled={artLinks.length <= 1}
                                onClick={() =>
                                  setArtLinks((rows) =>
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
                      </div>
                    )}

                    <div className="flex items-center justify-end gap-2">
                      <Button
                        type="button"
                        variant="ghost"
                        onClick={() => setShowForm(false)}
                      >
                        {tc.cancel ?? "Cancel"}
                      </Button>
                      <Button
                        type="submit"
                        disabled={isSavingArtifact || !artName.trim()}
                      >
                        {isSavingArtifact ? (
                          <Spinner />
                        ) : (
                          <Plus size={16} weight="bold" />
                        )}
                        {t.saveArtifact ?? "Save artifact"}
                      </Button>
                    </div>
                  </form>
                </Card>
              </Reveal>
            )}

            {/* Two-column: knowledge tree + main content */}
            <div className="mt-12 grid grid-cols-1 gap-8 lg:grid-cols-[240px_1fr]">
              <aside className="lg:sticky lg:top-20 lg:self-start">
                <Card className="p-3">
                  <KnowledgeTree
                    productId={product.id}
                    onSelect={onTreeSelect}
                    onMutate={fetchProduct}
                  />
                </Card>
              </aside>

              <div className="flex flex-col gap-12">
                {/* Artifacts */}
                <section>
                  <SectionHeader
                    title={t.artifactsTitle ?? "Artifacts"}
                    subtitle={
                      product.artifacts?.length
                        ? fmt(t.artifactsCount, { n: product.artifacts.length })
                        : (t.artifactsEmptySubtitle ?? "")
                    }
                  />

                  <div className="mt-6">
                    {!product.artifacts || product.artifacts.length === 0 ? (
                      <EmptyState
                        icon={<Plus size={20} weight="regular" />}
                        title={t.noArtifactsTitle ?? "No artifacts yet"}
                        description={t.noArtifactsDesc ?? ""}
                        action={
                          <Button onClick={() => setShowForm(true)}>
                            <Plus size={16} weight="bold" />
                            {t.addArtifact ?? "Add artifact"}
                          </Button>
                        }
                      />
                    ) : (
                      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                        {product.artifacts.map((art, i) => {
                          const meta = artifactTypeMeta(art.type);
                          const artLabel =
                            (tArt?.[art.type]?.label as string) ?? meta.label;
                          const isGenerating = generatingId === art.id;
                          const isDeleting = deletingArtId === art.id;
                          // Only codebase artifacts have generated docs to
                          // preview / (re)generate; spec & links are authored
                          // directly and rendered from `content`.
                          const isCodebase = art.type === "codebase";
                          const hasDocs = isCodebase && Boolean(art.generated_docs);
                          return (
                            <Reveal key={art.id} delayMs={Math.min(i, 6) * 80}>
                              <Card
                                hover
                                className="group relative flex h-full flex-col overflow-hidden p-6"
                              >
                                {isGenerating && isCodebase && (
                                  <span className="gen-progress-bar" aria-hidden />
                                )}
                                <div className="flex items-start justify-between gap-3">
                                  <div className="flex items-center gap-2.5">
                                    <span className="flex h-9 w-9 items-center justify-center rounded-md bg-surface-2 text-ink">
                                      {artifactIconFor(art.type)}
                                    </span>
                                    <div className="min-w-0">
                                      <h3 className="truncate text-sm font-medium text-ink">
                                        {art.name}
                                      </h3>
                                      <div className="mt-1 flex flex-wrap items-center gap-2">
                                        <Tag tone={meta.tone}>{artLabel}</Tag>
                                        {art.kind && (
                                          <Tag tone="neutral">{art.kind}</Tag>
                                        )}
                                        {hasDocs && (
                                          <Tag tone="green">{t.docsReady ?? "Docs ready"}</Tag>
                                        )}
                                        {art.verified && (
                                          <Tag tone="green">{t.verified ?? "Verified"}</Tag>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                  <IconButton
                                    aria-label={t.deleteArtifact ?? "Delete artifact"}
                                    title={t.deleteArtifact ?? "Delete artifact"}
                                    onClick={() => handleDeleteArtifact(art.id)}
                                    disabled={isDeleting}
                                    className="opacity-0 transition-opacity group-hover:opacity-100"
                                  >
                                    {isDeleting ? (
                                      <Spinner />
                                    ) : (
                                      <Trash size={16} weight="regular" />
                                    )}
                                  </IconButton>
                                </div>

                                {/* Type-specific meta */}
                                <div className="mt-4 space-y-1 text-xs text-muted">
                                  {isCodebase && art.repo_url && (
                                    <p className="truncate font-mono">
                                      {art.repo_url}
                                    </p>
                                  )}
                                  {art.content && (
                                    <p className="font-mono">
                                      {fmt(t.chars, { n: art.content.length })}
                                    </p>
                                  )}
                                </div>

                                {/* Generated docs preview (codebase only) */}
                                {hasDocs && (
                                  <div className="mt-4 max-h-28 overflow-hidden rounded-md border border-divider bg-surface-2 p-3 font-mono text-xs leading-relaxed text-muted">
                                    {art.generated_docs?.slice(0, 280)}
                                    {(art.generated_docs?.length ?? 0) > 280 &&
                                      "…"}
                                  </div>
                                )}

                                {/* Actions */}
                                <div className="mt-6 flex items-center justify-between border-t border-divider pt-4">
                                  <button
                                    onClick={() =>
                                      router.push(
                                        `/products/${product.id}/artifacts/${art.id}`,
                                      )
                                    }
                                    className={cn(
                                      "inline-flex items-center gap-1 text-xs font-medium text-ink",
                                      "transition-transform hover:translate-x-0.5",
                                    )}
                                  >
                                    {t.openDocs ?? "Open"}
                                    <ArrowRight size={14} weight="bold" />
                                  </button>
                                  {isCodebase && (
                                    <Button
                                      size="sm"
                                      variant="subtle"
                                      onClick={() => handleGenerate(art.id)}
                                      disabled={isGenerating}
                                    >
                                      {isGenerating ? (
                                        <Spinner />
                                      ) : (
                                        <Lightning size={14} weight="fill" />
                                      )}
                                      {isGenerating ? (t.generating ?? "Generating…") : (t.generate ?? "Generate")}
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

                {/* Expert agent chat (replaces Long-context RLM panel) */}
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
              </div>
            </div>
          </>
        ) : null}
      </main>
    </div>
  );
}
