"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Briefcase,
  Plus,
  StackSimple,
  Trash,
} from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import {
  Button,
  Card,
  EmptyState,
  IconButton,
  Input,
  Label,
  Reveal,
  SectionHeader,
  Spinner,
  Textarea,
  cn,
} from "@/components/ui";
import {
  ApiAbortError,
  ApiError,
  deleteJson,
  getJson,
  onApiError,
  postJson,
} from "@/lib/api";
import { type Product, type ProductListItem, generateId } from "@/lib/types";

export default function ProductsDashboard() {
  const router = useRouter();
  const { notify } = useNotifications();
  const { messages, fmt } = useLanguage();
  const t = messages?.home ?? {};
  const tc = messages?.common ?? {};
  // P1-16: GET /api/products serves light rows (counters only); a freshly
  // created full Product (POST response) is structurally assignable too —
  // hence the Partial<Product> widening (child arrays for the fallbacks).
  const [products, setProducts] = useState<
    (ProductListItem & Partial<Product>)[]
  >([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // SPEC-003: backend unreachable/degraded — quiet inline chip, no banner.
  const [degraded, setDegraded] = useState(false);
  const fetchAbortRef = useRef<AbortController | null>(null);

  // Create-product form state
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchProducts = useCallback(
    async (silent = false) => {
      // Supersede any in-flight list fetch (fresh data wins over stale).
      fetchAbortRef.current?.abort();
      const abort = new AbortController();
      fetchAbortRef.current = abort;
      if (!silent) setIsLoading(true);
      try {
        const data = await getJson<(ProductListItem & Partial<Product>)[]>(
          "/api/products",
          { signal: abort.signal },
        );
        setProducts(data);
        setError(null);
        setDegraded(false);
      } catch (e) {
        if (e instanceof ApiAbortError) return; // unmount/navigation — silent
        if (onApiError(e, router)) return;
        if (e instanceof ApiError && e.unavailable) {
          // Network/5xx after client retries: keep whatever data we have;
          // the recheck loop below reconnects once the backend is back.
          setDegraded(true);
          return;
        }
        setError(e instanceof Error ? e.message : "Failed to load products.");
      } finally {
        if (!silent) setIsLoading(false);
      }
    },
    [router],
  );

  useEffect(() => {
    fetchProducts();
    return () => fetchAbortRef.current?.abort();
  }, [fetchProducts]);

  // SPEC-003: while degraded, silently re-check every 5 s so a short backend
  // outage recovers on its own (chained, not interval — no overlapping calls).
  useEffect(() => {
    if (!degraded) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const recheck = async () => {
      await fetchProducts(true);
      if (alive) timer = setTimeout(recheck, 5_000);
    };
    timer = setTimeout(recheck, 5_000);
    return () => {
      alive = false;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [degraded, fetchProducts]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || isCreating) return;
    setIsCreating(true);
    try {
      const draft: Product = {
        id: generateId("prod"),
        name: name.trim(),
        description: description.trim(),
        codebases: [],
        specs: [],
        links: [],
      };
      // Non-GET failures surface normally via notify (SPEC-003).
      const saved = await postJson<Product>("/api/products", draft);
      setProducts((prev) => [saved, ...prev]);
      setName("");
      setDescription("");
      setShowForm(false);
    } catch (e) {
      if (onApiError(e, router)) return;
      notify({
        tone: "error",
        title: t.createFailedTitle ?? "Failed to create product",
        message: e instanceof Error ? e.message : "Failed to create product.",
      });
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(t.deleteConfirm ?? "Delete this product and all its artifacts?")) return;
    setDeletingId(id);
    try {
      await deleteJson(`/api/products/${id}`);
      setProducts((prev) => prev.filter((p) => p.id !== id));
    } catch (e) {
      if (onApiError(e, router)) return;
      notify({
        tone: "error",
        title: t.deleteFailedTitle ?? "Failed to delete product",
        message: e instanceof Error ? e.message : "Failed to delete product.",
      });
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <AppHeader />

      <main className="mx-auto  px-6 py-24">
        {/* Hero */}
        <Reveal>
          <SectionHeader
            title={t.title ?? "Products"}
            subtitle={t.subtitle ?? ""}
            action={
              <Button
                onClick={() => setShowForm((v) => !v)}
                variant={showForm ? "ghost" : "primary"}
              >
                <Plus size={16} weight="bold" />
                {showForm ? tc.close ?? "Close" : t.newProduct ?? "New product"}
              </Button>
            }
          />
        </Reveal>

        {/* Inline create form */}
        {showForm && (
          <Reveal className="mt-8">
            <Card className="p-6">
              <form onSubmit={handleCreate} className="grid gap-5">
                <div>
                  <Label>{t.name ?? "Name"}</Label>
                  <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder={t.namePlaceholder ?? "e.g. Payments Platform"}
                    required
                    autoFocus
                  />
                </div>
                <div>
                  <Label>{t.description ?? "Description"}</Label>
                  <Textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder={t.descPlaceholder ?? ""}
                    rows={3}
                    className="font-sans"
                  />
                </div>
                <div className="flex items-center justify-end gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => setShowForm(false)}
                  >
                    {tc.cancel ?? "Cancel"}
                  </Button>
                  <Button type="submit" disabled={isCreating || !name.trim()}>
                    {isCreating ? (
                      <Spinner />
                    ) : (
                      <Plus size={16} weight="bold" />
                    )}
                    {t.createProduct ?? "Create product"}
                  </Button>
                </div>
              </form>
            </Card>
          </Reveal>
        )}

        {/* Error banner */}
        {error && (
          <div className="mt-8 rounded-md border border-tag-red-bg bg-tag-red-bg px-4 py-3 text-sm text-tag-red-fg">
            {error}
          </div>
        )}

        {/* Quiet degraded indicator (SPEC-003): non-blocking, auto-recovers */}
        {degraded && !error && (
          <div className="mt-8 inline-flex items-center gap-2 rounded-md border border-divider bg-surface px-3 py-2 text-xs text-muted">
            <Spinner />
            {t.degraded ?? "Backend unavailable — retrying…"}
          </div>
        )}

        {/* Content */}
        <div className="mt-12">
          {isLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted">
              <Spinner /> {t.loadingProducts ?? "Loading products…"}
            </div>
          ) : degraded ? null : products.length === 0 ? ( // no false "no products" while backend is down
            <EmptyState
              icon={<Briefcase size={20} weight="regular" />}
              title={t.noProductsTitle ?? "No products yet"}
              description={t.noProductsDesc ?? ""}
              action={
                <Button onClick={() => setShowForm(true)}>
                  <Plus size={16} weight="bold" />
                  {t.createFirst ?? "Create your first product"}
                </Button>
              }
            />
          ) : (
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
              {products.map((p, i) => {
                // P2-33: databases are artifacts too — include them in the
                // dashboard counter. P1-16: list rows carry SQL-counted
                // totals; a full Product (POST-create prepend) carries child
                // arrays instead — hence the ?? fallbacks.
                const count =
                  (p.codebases_count ?? p.codebases?.length ?? 0) +
                  (p.specs_count ?? p.specs?.length ?? 0) +
                  (p.links_count ?? p.links?.length ?? 0) +
                  (p.databases_count ?? p.databases?.length ?? 0);
                const isDeleting = deletingId === p.id;
                return (
                  <Reveal key={p.id} delayMs={Math.min(i, 6) * 80}>
                    <Card hover className="group flex h-full flex-col p-6">
                      <div className="flex items-start justify-between gap-3">
                        <span className="text-xs text-muted">
                          {fmt(t.artifactCount, { n: count })}
                        </span>
                        <IconButton
                          aria-label={t.deleteTitle ?? "Delete product"}
                          title={t.deleteTitle ?? "Delete product"}
                          onClick={(e) => {
                            e.preventDefault();
                            handleDelete(p.id);
                          }}
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

                      <button
                        onClick={() => router.push(`/products/${p.id}`)}
                        className="mt-5 flex flex-1 flex-col items-start text-left"
                      >
                        <h3 className="font-editorial text-lg leading-snug tracking-tight text-ink">
                          {p.name}
                        </h3>
                        <p className="mt-2 line-clamp-3 text-[15px] text-muted">
                          {p.description || (t.noDescription ?? "No description yet.")}
                        </p>
                        {p.summary && (
                          <p className="mt-3 line-clamp-2 text-[13px] text-muted">
                            {p.summary.replace(/[#*`>_-]/g, "").slice(0, 160)}
                          </p>
                        )}
                      </button>

                      <div className="mt-6 flex items-center justify-between border-t border-divider pt-4">
                        <span className="font-mono text-xs text-muted">
                          {p.id}
                        </span>
                        <Link
                          href={`/products/${p.id}`}
                          className={cn(
                            "inline-flex items-center gap-1 text-xs font-medium text-ink",
                            "transition-transform hover:translate-x-0.5",
                          )}
                        >
                          {t.open ?? "Open"}
                          <ArrowRight size={14} weight="bold" />
                        </Link>
                      </div>
                    </Card>
                  </Reveal>
                );
              })}
            </div>
          )}
        </div>

        {/* Quiet footer note */}
        <div className="mt-24 flex items-center gap-2 text-xs text-muted">
          <StackSimple size={14} weight="regular" />
          {t.footer ?? "Documentation is generated locally — no cloud API keys required."}
        </div>
      </main>
    </div>
  );
}
