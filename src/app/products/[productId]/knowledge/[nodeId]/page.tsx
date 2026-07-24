"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  FileText,
  Spinner,
  UploadSimple,
} from "@phosphor-icons/react";
import { AppHeader } from "@/components/AppHeader";
import { MarkdownEditor, EditorSaveBar } from "@/components/MarkdownEditor";
import { KnowledgeTree } from "@/components/knowledge/KnowledgeTree";
import { useNotifications } from "@/contexts/NotificationContext";
import { VerifiedBadge, VerifiedButton } from "@/components/VerifiedBadge";
import {
  Button,
  Card,
  EmptyState,
  Reveal,
  SectionHeader,
  Spinner as SpinnerIcon,
  Tag,
} from "@/components/ui";
import { type KnowledgeNode, type Product } from "@/lib/types";
import { useLanguage } from "@/contexts/LanguageContext";

export default function KnowledgeNodePage() {
  const router = useRouter();
  const params = useParams<{ productId: string; nodeId: string }>();
  const { productId, nodeId } = params;
  const { notify } = useNotifications();
  const { messages, fmt } = useLanguage();
  const t = messages?.knowledge ?? {};

  const [node, setNode] = useState<KnowledgeNode | null>(null);
  const [product, setProduct] = useState<Product | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [content, setContent] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [verified, setVerified] = useState(false);
  const [verifiedBy, setVerifiedBy] = useState<string | null>(null);

  const fileRef = useRef<HTMLInputElement | null>(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nodeRes, prodRes] = await Promise.all([
        fetch(`/api/products/${productId}/knowledge/nodes/${nodeId}`, {
          credentials: "include",
          cache: "no-store",
        }),
        fetch(`/api/products/${productId}`, {
          credentials: "include",
          cache: "no-store",
        }),
      ]);
      if (nodeRes.status === 401) {
        router.replace(
          `/login?next=/products/${productId}/knowledge/${nodeId}`,
        );
        return;
      }
      if (!nodeRes.ok) {
        const msg = fmt(t.treeUnavailable ?? "Knowledge node unavailable ({status})", { status: String(nodeRes.status) });
        setError(msg);
        return;
      }
      const n = (await nodeRes.json()) as KnowledgeNode;
      setNode(n);
      setContent(n.content_md ?? "");
      setDirty(false);
      setVerified(Boolean(n.verified));
      setVerifiedBy(n.verified_by ?? null);
      if (prodRes.ok) {
        setProduct((await prodRes.json()) as Product);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : (t.loadFailedMessage ?? "Failed to load node.");
      setError(msg);
      notify({ tone: "error", title: t.loadFailedTitle ?? "Failed to load node", message: msg });
    } finally {
      setLoading(false);
    }
  }, [productId, nodeId, router, notify, t, fmt]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const save = async () => {
    if (!node || saving) return;
    setSaving(true);
    try {
      const res = await fetch(
        `/api/products/${productId}/knowledge/nodes/${nodeId}`,
        {
          method: "PUT",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content_md: content }),
        },
      );
      if (res.status === 401) {
        router.replace(
          `/login?next=/products/${productId}/knowledge/${nodeId}`,
        );
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Save failed (${res.status})`);
      }
      const updated = (await res.json()) as KnowledgeNode;
      setNode(updated);
      setDirty(false);
      notify({ tone: "success", title: t.pageSaved ?? "Page saved" });
    } catch (e) {
      notify({
        tone: "error",
        title: t.saveFailedTitle ?? "Save failed",
        message: e instanceof Error ? e.message : (t.saveFailedTitle ?? "Save failed"),
      });
    } finally {
      setSaving(false);
    }
  };

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !node || uploading) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(
        `/api/products/${productId}/knowledge/nodes/${nodeId}/upload`,
        {
          method: "POST",
          credentials: "include",
          body: fd,
        },
      );
      if (res.status === 401) {
        router.replace(
          `/login?next=/products/${productId}/knowledge/${nodeId}`,
        );
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Upload failed (${res.status})`);
      }
      const data = await res.json().catch(() => ({}));
      const md: string =
        data?.content_md ?? data?.markdown ?? data?.content ?? "";
      if (md) {
        setContent((prev) => (prev ? `${prev}\n\n${md}` : md));
        setDirty(true);
        notify({
          tone: "success",
          title: t.importedTitle ?? "Imported file",
          message: fmt(t.importedMessage ?? "", { file: file.name, n: String(md.length) }),
        });
      } else {
        notify({
          tone: "warning",
          title: t.uploadNoMarkdownTitle ?? "Upload succeeded",
          message: t.uploadNoMarkdown ?? "No markdown was returned.",
        });
      }
    } catch (err) {
      notify({
        tone: "error",
        title: t.uploadFailedTitle ?? "Upload failed",
        message: err instanceof Error ? err.message : (t.uploadFailedTitle ?? "Upload failed"),
      });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-canvas text-ink">
        <AppHeader />
        <div className="mt-24 flex items-center justify-center gap-2 text-sm text-muted">
          <Spinner /> {t.loadingNode ?? "Loading knowledge node…"}
        </div>
      </div>
    );
  }

  if (error || !node) {
    return (
      <div className="min-h-screen bg-canvas text-ink">
        <AppHeader />
        <main className="mx-auto max-w-3xl px-6 py-16">
          <EmptyState
            icon={<FileText size={20} weight="regular" />}
            title={t.notAvailable ?? "Knowledge node not available"}
            description={error || (t.notFoundFallback ?? "This node could not be found.")}
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
                <Tag tone="neutral">{node.node_type}</Tag>
                {node.node_type === "page" && (
                  <span className="font-mono text-xs text-muted">
                    {node.slug}
                  </span>
                )}
                {verified && (
                  <VerifiedBadge verified={verified} verifiedBy={verifiedBy} />
                )}
              </div>
              <h1 className="mt-2 font-editorial text-2xl tracking-tight text-ink">
                {node.title}
              </h1>
            </div>
            {node.node_type === "page" && (
              <VerifiedButton
                verified={verified}
                verifyUrl={`/api/products/${productId}/knowledge/nodes/${nodeId}/verify`}
                ownerId={product?.owner_id ?? node.created_by ?? null}
                onVerified={(next) => {
                  setVerified(next.verified);
                  setVerifiedBy(next.verified_by ?? null);
                }}
              />
            )}
          </div>
        </Reveal>

        {/* Two-column: tree + editor */}
        <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[240px_1fr]">
          <aside className="lg:sticky lg:top-20 lg:self-start">
            <Card className="p-3">
              <KnowledgeTree
                productId={productId}
                selectedNodeId={nodeId}
                onSelect={(n) => {
                  if (n.node_type === "page") {
                    router.push(`/products/${productId}/knowledge/${n.id}`);
                  }
                }}
                onMutate={fetchAll}
              />
            </Card>
          </aside>

          <div className="flex flex-col gap-6">
            {node.node_type === "page" ? (
              <>
                <SectionHeader
                  title={t.contentTitle ?? "Content"}
                  subtitle={t.contentSubtitle ?? ""}
                  action={
                    <div className="flex items-center gap-2">
                      <input
                        ref={fileRef}
                        type="file"
                        className="hidden"
                        accept=".docx,.pdf,.pptx,.html,.htm,.xlsx,.csv,.txt,.md,.json,.yaml,.yml"
                        onChange={onUpload}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => fileRef.current?.click()}
                        disabled={uploading}
                        title={t.uploadTitle ?? ""}
                      >
                        {uploading ? (
                          <SpinnerIcon />
                        ) : (
                          <UploadSimple size={14} weight="regular" />
                        )}
                        {uploading ? (t.converting ?? "Converting…") : (t.uploadFile ?? "Upload file")}
                      </Button>
                    </div>
                  }
                />

                <MarkdownEditor
                  value={content}
                  onChange={(v) => {
                    setContent(v);
                    setDirty(true);
                  }}
                />

                <EditorSaveBar
                  saving={saving}
                  dirty={dirty}
                  onSave={save}
                  saveLabel={t.savePage ?? "Save page"}
                />
              </>
            ) : (
              <EmptyState
                icon={<FileText size={20} weight="regular" />}
                title={t.folderTitle ?? "Folder"}
                description={t.containerDesc ?? ""}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
