"use client";

/**
 * SpecEditor — dedicated screen for creating and editing OpenAPI / AsyncAPI
 * specs (separate from the generic artifact viewer).
 *
 * Modes follow the knowledge-page pattern:
 *  - Existing spec: opens in preview-only view; an Edit button toggles
 *    editing (default split), with a switch to a pure editor pane.
 *  - New spec: opens straight in editing (default split).
 *
 * Save:
 *  - new  → POST /api/products/{id}/specs            (body: {id,name,kind,content,source})
 *  - edit → PUT  /api/products/{id}/specs/{specId}   (body: {content}; name/kind read-only)
 *
 * Delete: DELETE /api/products/{id}/specs/{specId}, redirect to product page.
 * Verify: POST /api/products/{id}/specs/{specId}/verify (owner/admin only).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Eye,
  FileText,
  PencilSimple,
  Trash,
} from "@phosphor-icons/react";
import { ArrowsOutSimple } from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { VerifiedBadge, VerifiedButton } from "@/components/VerifiedBadge";
import {
  Button,
  Card,
  EmptyState,
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
  type Product,
  type Spec,
  type SpecKind,
  generateId,
} from "@/lib/types";
import { SpecViewer } from "@/components/SpecViewer";

type View = "edit" | "split" | "preview";

interface SpecEditorProps {
  productId: string;
  /** Spec id to edit/view. When null, the component creates a new spec. */
  specId?: string | null;
}

export function SpecEditor({ productId, specId }: SpecEditorProps) {
  const router = useRouter();
  const { notify } = useNotifications();
  const { messages } = useLanguage();
  const t = messages?.specEditor ?? {};
  const tArt = messages?.artifactTypes ?? {};
  const tc = messages?.common ?? {};

  const isNew = !specId;

  const [product, setProduct] = useState<Product | null>(null);
  const [spec, setSpec] = useState<Spec | null>(null);
  const [loading, setLoading] = useState(!isNew);
  const [error, setError] = useState<string | null>(null);

  // Editor draft fields. For new specs name/kind are editable; for existing
  // they are read-only (the PUT contract only updates content).
  const [name, setName] = useState("");
  const [kind, setKind] = useState<SpecKind>("openapi");
  const [content, setContent] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [verified, setVerified] = useState(false);
  const [verifiedBy, setVerifiedBy] = useState<string | null>(null);

  // New specs edit immediately; existing specs open in preview-only view and
  // switch to editing (split) when the user clicks Edit.
  const [editing, setEditing] = useState(isNew);
  const [view, setView] = useState<View>(isNew ? "split" : "preview");

  const fetchAll = useCallback(async () => {
    if (isNew) return;
    setLoading(true);
    setError(null);
    try {
      const prodRes = await fetch(`/api/products/${productId}`, {
        credentials: "include",
        cache: "no-store",
      });
      if (prodRes.status === 401) {
        router.replace(`/login?next=/products/${productId}/specs/${specId}`);
        return;
      }
      if (prodRes.status === 404) {
        setError(t.productNotFound ?? "Product not found.");
        return;
      }
      const prod = (await prodRes.json()) as Product;
      setProduct(prod);
      const found = prod.specs.find((s) => s.id === specId) ?? null;
      if (!found) {
        setError(t.notFoundFallback ?? "This spec could not be found.");
        return;
      }
      setSpec(found);
      setContent(found.content ?? "");
      setDirty(false);
      setVerified(Boolean(found.verified));
      setVerifiedBy(found.verified_by ?? null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : (t.loadFailedTitle ?? "Failed to load.");
      setError(msg);
      notify({ tone: "error", title: t.loadFailedTitle ?? "Failed to load", message: msg });
    } finally {
      setLoading(false);
    }
  }, [isNew, productId, specId, router, notify, t, messages]);

  useEffect(() => {
    if (isNew) return;
    fetchAll();
  }, [fetchAll, isNew]);

  const save = async () => {
    if (saving) return;
    if (isNew && !name.trim()) return;
    setSaving(true);
    try {
      if (isNew) {
        const draft: Spec = {
          id: generateId("spec"),
          name: name.trim(),
          kind,
          content: content || null,
          source: "manual",
        };
        const res = await fetch(`/api/products/${productId}/specs`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(draft),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.detail || `Create failed (${res.status})`);
        }
        const updated = (await res.json()) as Product;
        const saved = updated.specs.find((s) => s.id === draft.id) ?? null;
        notify({ tone: "success", title: t.savedTitle ?? "Saved", message: t.savedMessage ?? "" });
        // Switch into edit-view of the freshly created spec.
        router.replace(`/products/${productId}/specs/${draft.id}`);
        setProduct(updated);
        setSpec(saved);
        setDirty(false);
        setEditing(false);
        setView("preview");
        return;
      }
      // Existing: PUT content only.
      const res = await fetch(
        `/api/products/${productId}/specs/${specId}`,
        {
          method: "PUT",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
        },
      );
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}/specs/${specId}`);
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Save failed (${res.status})`);
      }
      const updated = (await res.json()) as Product;
      setProduct(updated);
      const found = updated.specs.find((s) => s.id === specId) ?? null;
      setSpec(found);
      setDirty(false);
      setEditing(false);
      setView("preview");
      notify({ tone: "success", title: t.savedTitle ?? "Saved", message: t.savedMessage ?? "" });
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

  const remove = async () => {
    if (!spec || deleting) return;
    if (!confirm(t.deleteConfirm ?? "Delete this spec?")) return;
    setDeleting(true);
    try {
      const res = await fetch(
        `/api/products/${productId}/specs/${specId}`,
        { method: "DELETE", credentials: "include" },
      );
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}/specs/${specId}`);
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

  const viewBtns: { key: View; labelKey: "edit" | "split" | "preview"; icon: typeof Eye }[] = [
    { key: "edit", labelKey: "edit", icon: PencilSimple },
    { key: "split", labelKey: "split", icon: ArrowsOutSimple },
    { key: "preview", labelKey: "preview", icon: Eye },
  ];

  const header = useMemo(() => {
    const label = isNew ? (t.newSpecTitle ?? "New spec") : spec?.name ?? (t.specTitle ?? "Spec");
    return label;
  }, [isNew, spec, t]);

  if (loading) {
    return (
      <div className="min-h-screen bg-canvas text-ink">
        <AppHeader />
        <div className="mt-24 flex items-center justify-center gap-2 text-sm text-muted">
          <Spinner /> {t.loading ?? "Loading…"}
        </div>
      </div>
    );
  }

  if (error || (!isNew && !spec)) {
    return (
      <div className="min-h-screen bg-canvas text-ink">
        <AppHeader />
        <main className="mx-auto max-w-3xl px-6 py-16">
          <EmptyState
            icon={<FileText size={20} weight="regular" />}
            title={t.notAvailable ?? "Spec not available"}
            description={error || (t.notFoundFallback ?? "")}
            action={
              <Link href={`/products/${productId}`}>
                <Button>{t.backToProduct ?? "Back to product"}</Button>
              </Link>
            }
          />
        </main>
      </div>
    );
  }

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

        {/* Header */}
        <Reveal className="mt-6">
          <div className="flex flex-col gap-4 border-b border-divider pb-6 md:flex-row md:items-center md:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Tag tone="green">{tArt?.spec?.label ?? "Spec"}</Tag>
                {(isNew ? kind : spec?.kind) && (
                  <Tag tone="neutral">{(isNew ? kind : spec?.kind) ?? ""}</Tag>
                )}
                {!isNew && verified && (
                  <VerifiedBadge verified={verified} verifiedBy={verifiedBy} />
                )}
              </div>
              <h1 className="mt-2 font-editorial text-2xl tracking-tight text-ink">
                {header}
              </h1>
            </div>
            <div className="flex items-center gap-2">
              {!isNew && !editing && (
                <Button
                  type="button"
                  variant="subtle"
                  size="sm"
                  onClick={() => {
                    setEditing(true);
                    setView("split");
                  }}
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
                    setView("preview");
                    // Discard unsaved draft back to the persisted content.
                    if (spec) setContent(spec.content ?? "");
                    setDirty(false);
                  }}
                  disabled={saving}
                >
                  {tc.cancel ?? "Cancel"}
                </Button>
              )}
              {!isNew && (
                <>
                  <VerifiedButton
                    verified={verified}
                    verifyUrl={`/api/products/${productId}/specs/${specId}/verify`}
                    ownerId={product?.owner_id ?? null}
                    onVerified={(next) => {
                      setVerified(next.verified);
                      setVerifiedBy(next.verified_by ?? null);
                    }}
                  />
                  <Button
                    type="button"
                    variant="danger"
                    size="sm"
                    onClick={remove}
                    disabled={deleting}
                  >
                    {deleting ? <Spinner /> : <Trash size={14} weight="regular" />}
                    {t.delete ?? "Delete"}
                  </Button>
                </>
              )}
            </div>
          </div>
        </Reveal>

        {/* Create-only metadata fields */}
        {isNew && (
          <Reveal className="mt-8">
            <Card className="p-6">
              <div className="grid gap-5 md:grid-cols-[1fr_220px]">
                <div>
                  <Label>{t.specName ?? "Name"}</Label>
                  <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder={t.specNamePlaceholder ?? "e.g. Checkout API"}
                    required
                    autoFocus
                  />
                </div>
                <div>
                  <Label>{t.specKind ?? "Spec kind"}</Label>
                  <Select
                    value={kind}
                    onChange={(e) => setKind(e.target.value as SpecKind)}
                  >
                    <option value="openapi">{t.openapi ?? "OpenAPI (REST)"}</option>
                    <option value="asyncapi">{t.asyncapi ?? "AsyncAPI (events)"}</option>
                  </Select>
                </div>
              </div>
            </Card>
          </Reveal>
        )}

        {/* Editor / preview */}
        <Reveal className="mt-8">
          <Card className="overflow-hidden">
            <SectionHeader
              title={t.contentTitle ?? "Specification"}
              className="px-4 pt-4"
              action={
                editing ? (
                  <div className="flex items-center gap-0.5">
                    {viewBtns.map((v) => (
                      <button
                        key={v.key}
                        type="button"
                        onClick={() => setView(v.key)}
                        className={cn(
                          "inline-flex h-8 items-center gap-1 rounded px-2 text-xs font-medium transition-colors",
                          view === v.key
                            ? "bg-surface-2 text-ink"
                            : "text-muted hover:bg-surface-2 hover:text-ink",
                        )}
                      >
                        <v.icon size={13} weight="regular" />
                        <span className="hidden sm:inline">
                          {t[v.labelKey] ?? v.labelKey}
                        </span>
                      </button>
                    ))}
                  </div>
                ) : undefined
              }
            />

            {/* View (read-only): SpecViewer */}
            {!editing ? (
              <div className="px-4 pb-6 pt-2">
                {content.trim() ? (
                  <SpecViewer content={content} kind={spec?.kind} />
                ) : (
                  <EmptyState
                    icon={<FileText size={20} weight="regular" />}
                    title={t.noContent ?? "No content yet"}
                    description={t.noContentDesc ?? "Add the spec content from the edit button."}
                    action={
                      <Button
                        variant="subtle"
                        size="sm"
                        onClick={() => {
                          setEditing(true);
                          setView("split");
                        }}
                      >
                        <PencilSimple size={14} weight="regular" />
                        {t.edit ?? "Edit"}
                      </Button>
                    }
                  />
                )}
              </div>
            ) : (
              <>
                <div className="grid md:grid-cols-2">
                  {(view === "edit" || view === "split") && (
                    <Textarea
                      value={content}
                      onChange={(e) => {
                        setContent(e.target.value);
                        setDirty(true);
                      }}
                      placeholder={t.specPlaceholder ?? "Paste your OpenAPI or AsyncAPI JSON/YAML here…"}
                      rows={20}
                      className={cn(
                        "min-h-[420px] w-full resize-y rounded-none border-0 border-divider focus:border-0 focus:outline-none",
                        view === "split" && "md:border-r",
                      )}
                    />
                  )}
                  {(view === "preview" || view === "split") && (
                    <div className="min-h-[420px] overflow-auto px-4 py-3">
                      {content.trim() ? (
                        <SpecViewer content={content} kind={isNew ? kind : spec?.kind} />
                      ) : (
                        <p className="text-sm text-muted">
                          {t.nothingToPreview ?? "Nothing to preview yet."}
                        </p>
                      )}
                    </div>
                  )}
                </div>
                {/* Save bar */}
                <div className="flex items-center justify-end gap-2 border-t border-divider px-4 py-3">
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => {
                      setEditing(false);
                      setView("preview");
                      if (spec) setContent(spec.content ?? "");
                      setDirty(false);
                    }}
                    disabled={saving}
                  >
                    {tc.cancel ?? "Cancel"}
                  </Button>
                  <Button
                    type="button"
                    onClick={save}
                    disabled={saving || !dirty || (isNew && !name.trim())}
                  >
                    {saving ? <Spinner /> : <PencilSimple size={14} weight="regular" />}
                    {saving ? (tc.saving ?? "Saving…") : (tc.save ?? "Save")}
                  </Button>
                </div>
              </>
            )}
          </Card>
        </Reveal>
      </main>
    </div>
  );
}

export default SpecEditor;
