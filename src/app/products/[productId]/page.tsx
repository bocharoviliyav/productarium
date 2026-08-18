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
  FileText,
  GitBranch,
  Lightning,
  LinkSimple,
  PencilSimple,
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
  type Product,
  type Spec,
  entityPath,
  generateId,
  parseLinksContent,
  serializeLinksContent,
} from "@/lib/types";
import { useNotifications } from "@/contexts/NotificationContext";

type DeleteType = "codebase" | "spec" | "links";

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
  const [isSaving, setIsSaving] = useState(false);

  // In-place link add form (inside the Links spoiler).
  const [addLinkOpen, setAddLinkOpen] = useState(false);
  const [linkName, setLinkName] = useState("");
  const [linkUrl, setLinkUrl] = useState("");
  const [linkDesc, setLinkDesc] = useState("");

  const [generatingId, setGeneratingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [linksOpen, setLinksOpen] = useState(false);

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

  const resetCodebaseForm = () => {
    setCbName("");
    setCbRepoUrl("");
    setCbRepoType("github");
  };

  const resetLinkForm = () => {
    setLinkName("");
    setLinkUrl("");
    setLinkDesc("");
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
        source: "manual" as const,
      };
      const res = await fetch(`/api/products/${product.id}/codebases`, {
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
      const res = await fetch(`/api/products/${product.id}/links`, {
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
      const res = await fetch(
        `/api/products/${product.id}/${type}/${entityId}`,
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

  const handleGenerate = async (type: "codebase" | "spec", entityId: string) => {
    if (!product) return;
    setGeneratingId(entityId);
    setError(null);
    try {
      // The FastAPI generate routes are registered under the PLURAL segment
      // (codebases / specs), but `type` is the singular form-state value
      // (codebase / spec). Interpolating it raw produces a 404; route the
      // segment through entityPath() (see src/lib/types.ts).
      const res = await fetch(
        `/api/products/${product.id}/${entityPath(type)}/${entityId}/generate`,
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
      const jobId = data.job_id;
      if (!jobId) {
        notify({ tone: "info", title: t.genTitle ?? "Generation", message: data.message || data.status || (t.genTriggered ?? "Generation triggered.") });
        await fetchProduct();
        return;
      }
      notify({ tone: "info", title: t.genTitle ?? "Generation", message: t.genStarted ?? "Generation started…" });
      const maxWaitMs = 30 * 60 * 1000;
      const startedAt = Date.now();
      while (Date.now() - startedAt < maxWaitMs) {
        if (generateAbortRef.current) return;
        await new Promise((r) => setTimeout(r, 2000));
        if (generateAbortRef.current) return;
        const stRes = await fetch(
          `/api/products/${product.id}/${entityPath(type)}/${entityId}/generate/status?job_id=${encodeURIComponent(jobId)}`,
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
      }
      throw new Error(t.genTimeout ?? "Generation timed out.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : (t.genFailed ?? "Generation failed.");
      notify({ tone: "error", title: t.genTitle ?? "Generation", message: msg });
    } finally {
      setGeneratingId(null);
    }
  };

  const onTreeSelect = (node: { node_type: string; id: string }) => {
    if (node.node_type === "page") {
      router.push(`/products/${productId}/knowledge/${node.id}`);
    }
  };

  const codebases = product?.codebases ?? [];
  const specs = product?.specs ?? [];
  const links = product?.links ?? [];

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
                          const isDeleting = deletingId === l.id;
                          return (
                            <li
                              key={l.id}
                              className="rounded-md border border-divider bg-surface-2 p-3"
                            >
                              <div className="flex items-center justify-between gap-2">
                                {firstUrl ? (
                                  <a
                                    href={firstUrl}
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
                                  {items.map((it, i) => (
                                    <li key={i} className="flex flex-col gap-0.5 text-sm">
                                      {it.url ? (
                                        <a
                                          href={it.url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="font-mono text-xs text-ink underline-offset-2 hover:underline"
                                        >
                                          {it.url}
                                        </a>
                                      ) : null}
                                      {it.description && (
                                        <span className="text-muted">{it.description}</span>
                                      )}
                                    </li>
                                  ))}
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
                          const isGenerating = generatingId === c.id;
                          const isDeleting = deletingId === c.id;
                          const hasDocs = Boolean(c.generated_docs);
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

                                {c.repo_url && (
                                  <p className="mt-4 truncate font-mono text-xs text-muted">
                                    {c.repo_url}
                                  </p>
                                )}

                                {hasDocs && (
                                  <div className="mt-4 max-h-28 overflow-hidden rounded-md border border-divider bg-surface-2 p-3 font-mono text-xs leading-relaxed text-muted">
                                    {c.generated_docs?.slice(0, 280)}
                                    {(c.generated_docs?.length ?? 0) > 280 && "…"}
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
                                  <Button
                                    size="sm"
                                    variant="subtle"
                                    onClick={() => handleGenerate("codebase", c.id)}
                                    disabled={isGenerating}
                                  >
                                    {isGenerating ? <Spinner /> : <Lightning size={14} weight="fill" />}
                                    {isGenerating ? (t.generating ?? "Generating…") : (t.generate ?? "Generate")}
                                  </Button>
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
              </div>
            </div>
          </>
        ) : null}
      </main>
    </div>
  );
}
