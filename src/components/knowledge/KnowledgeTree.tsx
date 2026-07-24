"use client";

/**
 * Confluence-like knowledge tree sidebar (item 2).
 *
 * Fetches GET /api/products/{id}/knowledge/tree and renders a nested,
 * expand/collapse tree of pages and folders only (no branch type). Supports
 * add (page/folder), rename, delete, and native HTML5 drag-and-drop to move a
 * node under another node (page or folder) or back to the root.
 *
 * Create/update/delete target POST/PUT/DELETE
 * /api/products/{id}/knowledge/nodes/{nodeId} — the client generates the node
 * id (matching the existing product/artifact create pattern where the client
 * supplies the id and the backend upserts). Moves (drag-and-drop) PUT
 * ``parent_id`` onto the moved node; the backend validates same-product + cycle
 * rules.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  CaretDown,
  CaretRight,
  DotsSixVertical,
  FilePlus,
  FolderPlus,
  PencilSimple,
  Plus,
  SealCheck,
  Spinner,
  Trash,
  TreeStructure,
} from "@phosphor-icons/react";
import {
  Button,
  cn,
  IconButton,
  Input,
  Label,
  Modal,
} from "@/components/ui";
import { useNotifications } from "@/contexts/NotificationContext";
import { useLanguage } from "@/contexts/LanguageContext";
import {
  type KnowledgeNode,
  type KnowledgeNodeType,
  buildKnowledgeTree,
  slugify,
} from "@/lib/types";

interface KnowledgeTreeProps {
  productId: string;
  selectedNodeId?: string | null;
  onSelect: (node: KnowledgeNode) => void;
  /** Called after any mutation so parents can refetch. */
  onMutate?: () => void;
  className?: string;
}

function nodeIcon(type: KnowledgeNodeType) {
  // Only page/folder exist now; folder gets the tree icon, page the file icon.
  if (type === "folder") return <TreeStructure size={14} weight="regular" />;
  return <FilePlus size={14} weight="regular" />;
}

/** Collect a node's own id + all descendant ids (for drop-cycle prevention). */
function collectDescendantIds(node: KnowledgeNode): string[] {
  const ids: string[] = [node.id];
  for (const child of node.children ?? []) {
    ids.push(...collectDescendantIds(child));
  }
  return ids;
}

interface TreeRowProps {
  node: KnowledgeNode;
  depth: number;
  selectedId?: string | null;
  onSelect: (n: KnowledgeNode) => void;
  onAddChild: (parent: KnowledgeNode) => void;
  onRename: (n: KnowledgeNode) => void;
  onDelete: (n: KnowledgeNode) => void;
  onDrop: (draggedId: string, targetParentId: string | null) => void;
  /** Flat lookup of node id -> node (across the whole tree) for cycle checks. */
  allById: Map<string, KnowledgeNode>;
}

function TreeRow({
  node,
  depth,
  selectedId,
  onSelect,
  onAddChild,
  onRename,
  onDelete,
  onDrop,
  allById,
}: TreeRowProps) {
  const [open, setOpen] = useState(true);
  const [isDropTarget, setIsDropTarget] = useState(false);
  const hasChildren = (node.children?.length ?? 0) > 0;
  const isActive = selectedId === node.id;
  const { messages } = useLanguage();
  const t = messages?.knowledge ?? {};

  // A drop is allowed when the dragged node is not this node and not one of
  // this node's descendants (prevents making a node its own ancestor).
  const canAcceptDrop = useCallback(
    (draggedId: string) => {
      if (draggedId === node.id) return false;
      const dragged = allById.get(draggedId);
      if (!dragged) return false;
      return !collectDescendantIds(dragged).includes(node.id);
    },
    [node.id, allById],
  );

  return (
    <div>
      <div
        className={cn(
          "group flex items-center gap-1 rounded-md py-1 pr-1 text-sm transition-colors",
          isActive
            ? "bg-surface-2 font-medium text-ink"
            : "text-muted hover:bg-surface-2 hover:text-ink",
          isDropTarget && "ring-2 ring-ink ring-inset",
        )}
        style={{ paddingLeft: depth * 12 + 4 }}
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData("text/plain", node.id);
          e.dataTransfer.effectAllowed = "move";
        }}
        onDragOver={(e) => {
          // Allow move cursor; cycle check happens in onDrop (the dragged id is
          // only readable inside drop, not during dragover).
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          setIsDropTarget(true);
        }}
        onDragLeave={() => setIsDropTarget(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDropTarget(false);
          const draggedId = e.dataTransfer.getData("text/plain");
          if (!draggedId || !canAcceptDrop(draggedId)) return;
          onDrop(draggedId, node.id);
        }}
      >
        <span
          className="inline-flex h-5 w-4 cursor-grab items-center justify-center text-muted opacity-0 transition-opacity group-hover:opacity-100"
          aria-hidden
          title={t.dragHandle ?? "Drag to move"}
        >
          <DotsSixVertical size={12} weight="bold" />
        </span>
        <button
          type="button"
          onClick={() => hasChildren && setOpen((v) => !v)}
          className={cn(
            "inline-flex h-5 w-5 items-center justify-center rounded text-muted",
            !hasChildren && "invisible",
          )}
          aria-label={open ? (t.collapse ?? "Collapse") : (t.expand ?? "Expand")}
        >
          {open ? <CaretDown size={12} weight="bold" /> : <CaretRight size={12} weight="bold" />}
        </button>
        <button
          type="button"
          onClick={() => onSelect(node)}
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
        >
          <span className="shrink-0 text-muted">{nodeIcon(node.node_type)}</span>
          <span className="truncate">{node.title}</span>
          {node.verified && (
            <SealCheck size={12} weight="fill" className="shrink-0 text-tag-green-fg" />
          )}
        </button>
        <span className="flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
          <IconButton
            aria-label={t.addChild ?? "Add page"}
            title={t.addChild ?? "Add page"}
            className="h-6 w-6"
            onClick={() => onAddChild(node)}
          >
            <Plus size={12} weight="bold" />
          </IconButton>
          <IconButton
            aria-label={t.rename ?? "Rename"}
            title={t.rename ?? "Rename"}
            className="h-6 w-6"
            onClick={() => onRename(node)}
          >
            <PencilSimple size={12} weight="regular" />
          </IconButton>
          <IconButton
            aria-label={t.deleteNode ?? "Delete"}
            title={t.deleteNode ?? "Delete"}
            className="h-6 w-6"
            onClick={() => onDelete(node)}
          >
            <Trash size={12} weight="regular" />
          </IconButton>
        </span>
      </div>
      {open && hasChildren && (
        <div>
          {node.children!.map((child) => (
            <TreeRow
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              onSelect={onSelect}
              onAddChild={onAddChild}
              onRename={onRename}
              onDelete={onDelete}
              onDrop={onDrop}
              allById={allById}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function KnowledgeTree({
  productId,
  selectedNodeId,
  onSelect,
  onMutate,
  className,
}: KnowledgeTreeProps) {
  const [nodes, setNodes] = useState<KnowledgeNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);

  const router = useRouter();
  const { notify } = useNotifications();
  const { messages, fmt } = useLanguage();
  const t = messages?.knowledge ?? {};
  const tRef = useRef(t);
  tRef.current = t;

  // Fatal fetch failure shown inline while loading (cleared once loaded).
  const [loadError, setLoadError] = useState<string | null>(null);

  // Add-node modal. The type is fixed by which button opened the modal, so the
  // modal no longer has a type selector (only page/folder are supported).
  const [addOpen, setAddOpen] = useState(false);
  const [addParent, setAddParent] = useState<KnowledgeNode | null>(null);
  const [addTitle, setAddTitle] = useState("");
  const [addType, setAddType] = useState<KnowledgeNodeType>("page");

  // Rename modal
  const [renameNode, setRenameNode] = useState<KnowledgeNode | null>(null);
  const [renameTitle, setRenameTitle] = useState("");

  // Drop-to-root indicator (the area above the tree clears parent_id).
  const [isRootDropTarget, setIsRootDropTarget] = useState(false);

  const fetchTree = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await fetch(`/api/products/${productId}/knowledge/tree`, {
        credentials: "include",
        cache: "no-store",
      });
      if (res.status === 401) {
        router.replace(`/login?next=/products/${productId}`);
        return;
      }
      if (!res.ok) {
        const msg = fmt(tRef.current.treeUnavailable ?? "Knowledge tree unavailable ({status})", { status: String(res.status) });
        notify({ tone: "error", title: tRef.current.treeTitle ?? "Knowledge tree", message: msg });
        setLoadError(msg);
        setNodes([]);
        return;
      }
      const data = (await res.json()) as KnowledgeNode[] | { nodes: KnowledgeNode[] };
      const flat = Array.isArray(data) ? data : data?.nodes ?? [];
      setNodes(flat);
    } catch (e) {
      const msg = e instanceof Error ? e.message : (tRef.current.loadFailedMessage ?? "Failed to load knowledge tree");
      notify({ tone: "error", title: tRef.current.treeTitle ?? "Knowledge tree", message: msg });
      setLoadError(msg);
      setNodes([]);
    } finally {
      setLoading(false);
    }
  }, [productId, router, notify, fmt]);

  useEffect(() => {
    void fetchTree();
  }, [fetchTree]);

  const tree = useMemo(() => {
    // If the response is already nested (nodes carry children), use it;
    // otherwise build from the flat parent_id list.
    const nested = nodes.some((n) => Array.isArray(n.children) && n.children.length > 0);
    return nested ? nodes : buildKnowledgeTree(nodes);
  }, [nodes]);

  // Flat id -> node lookup (with children attached) for drop-cycle checks.
  const allById = useMemo(() => {
    const map = new Map<string, KnowledgeNode>();
    const walk = (list: KnowledgeNode[]) => {
      for (const n of list) {
        map.set(n.id, n);
        if (n.children) walk(n.children);
      }
    };
    walk(tree);
    return map;
  }, [tree]);

  const openAdd = (parent: KnowledgeNode | null, type: KnowledgeNodeType) => {
    setAddParent(parent);
    setAddTitle("");
    setAddType(type);
    setAddOpen(true);
  };

  const submitAdd = async () => {
    const title = addTitle.trim();
    if (!title) return;
    setBusy("add");
    try {
      // The backend contract is POST /knowledge/nodes (NO id in the path);
      // the server generates the node id via _new_node_id() and takes
      // product_id from the URL. Sending a client id in the path matched no
      // route and returned 405 Method Not Allowed.
      const res = await fetch(
        `/api/products/${productId}/knowledge/nodes`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            parent_id: addParent?.id ?? null,
            title,
            slug: slugify(title),
            node_type: addType,
            content_md: addType === "page" ? "" : null,
            source: "manual",
          }),
        },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Create failed (${res.status})`);
      }
      setAddOpen(false);
      await fetchTree();
      onMutate?.();
    } catch (e) {
      notify({ tone: "error", title: t.treeTitle ?? "Knowledge tree", message: e instanceof Error ? e.message : (t.createFailed ?? "Create failed") });
    } finally {
      setBusy(null);
    }
  };

  const submitRename = async () => {
    if (!renameNode || !renameTitle.trim()) return;
    setBusy("rename");
    try {
      const res = await fetch(
        `/api/products/${productId}/knowledge/nodes/${renameNode.id}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            title: renameTitle.trim(),
            slug: slugify(renameTitle.trim()),
          }),
        },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Rename failed (${res.status})`);
      }
      setRenameNode(null);
      await fetchTree();
      onMutate?.();
    } catch (e) {
      notify({ tone: "error", title: t.treeTitle ?? "Knowledge tree", message: e instanceof Error ? e.message : (t.renameFailed ?? "Rename failed") });
    } finally {
      setBusy(null);
    }
  };

  const submitDelete = async (node: KnowledgeNode) => {
    if (!confirm(fmt(t.deleteConfirm ?? "Delete \u201c{name}\u201d and its subtree?", { name: node.title }))) return;
    setBusy(node.id);
    try {
      const res = await fetch(
        `/api/products/${productId}/knowledge/nodes/${node.id}`,
        { method: "DELETE", credentials: "include" },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Delete failed (${res.status})`);
      }
      await fetchTree();
      onMutate?.();
    } catch (e) {
      notify({ tone: "error", title: t.treeTitle ?? "Knowledge tree", message: e instanceof Error ? e.message : (t.deleteFailed ?? "Delete failed") });
    } finally {
      setBusy(null);
    }
  };

  // Drag-and-drop move: PUT parent_id onto the dragged node. targetParentId
  // null = move to root. The backend re-validates same-product + cycle.
  const submitMove = useCallback(
    async (draggedId: string, targetParentId: string | null) => {
      if (draggedId === targetParentId) return;
      setBusy(draggedId);
      try {
        const res = await fetch(
          `/api/products/${productId}/knowledge/nodes/${draggedId}`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({ parent_id: targetParentId }),
          },
        );
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.detail || `Move failed (${res.status})`);
        }
        await fetchTree();
        onMutate?.();
      } catch (e) {
        notify({
          tone: "error",
          title: t.treeTitle ?? "Knowledge tree",
          message: e instanceof Error ? e.message : (t.moveFailed ?? "Move failed"),
        });
      } finally {
        setBusy(null);
      }
    },
    [productId, fetchTree, onMutate, notify, t.treeTitle, t.moveFailed],
  );

  return (
    <div className={cn("flex flex-col", className)}>
      <div className="flex items-center justify-between px-1 pb-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
          {t.header ?? "Knowledge"}
        </h3>
        <div className="flex items-center gap-0.5">
          <IconButton
            aria-label={t.addPage ?? "Add page"}
            title={t.addPage ?? "Add page"}
            className="h-7 w-7"
            onClick={() => openAdd(null, "page")}
          >
            <FilePlus size={14} weight="regular" />
          </IconButton>
          <IconButton
            aria-label={t.addFolder ?? "Add folder"}
            title={t.addFolder ?? "Add folder"}
            className="h-7 w-7"
            onClick={() => openAdd(null, "folder")}
          >
            <FolderPlus size={14} weight="regular" />
          </IconButton>
        </div>
      </div>

      {loadError && (
        <p className="mb-2 rounded-md border border-tag-red-bg bg-tag-red-bg px-3 py-2 text-xs text-tag-red-fg">
          {loadError}
        </p>
      )}

      {/* Root drop zone: dropping a node here clears its parent_id. */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          setIsRootDropTarget(true);
        }}
        onDragLeave={() => setIsRootDropTarget(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsRootDropTarget(false);
          const draggedId = e.dataTransfer.getData("text/plain");
          if (!draggedId) return;
          submitMove(draggedId, null);
        }}
        className={cn(
          "mb-1 rounded-md border border-dashed border-divider px-2 py-1 text-center text-[10px] uppercase tracking-wide text-muted transition-colors",
          isRootDropTarget && "border-ink bg-surface-2 text-ink",
        )}
      >
        {t.rootDropZone ?? "Root — drop here to move to top level"}
      </div>

      <div className="min-h-[120px] flex-1 overflow-y-auto pr-1">
        {loading ? (
          <div className="flex items-center gap-2 px-2 py-4 text-xs text-muted">
            <Spinner /> {t.loadingTree ?? "Loading tree…"}
          </div>
        ) : tree.length === 0 ? (
          <div className="rounded-md border border-dashed border-divider bg-surface-2 px-3 py-6 text-center text-xs text-muted">
            {t.empty ?? "No knowledge pages yet."}
          </div>
        ) : (
          <nav className="flex flex-col gap-0.5">
            {tree.map((n) => (
              <TreeRow
                key={n.id}
                node={n}
                depth={0}
                selectedId={selectedNodeId}
                onSelect={onSelect}
                onAddChild={(parent) => openAdd(parent, "page")}
                onRename={(node) => {
                  setRenameNode(node);
                  setRenameTitle(node.title);
                }}
                onDelete={submitDelete}
                onDrop={submitMove}
                allById={allById}
              />
            ))}
          </nav>
        )}
      </div>

      {/* Add modal — type is fixed by the button, no selector shown. */}
      <Modal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        title={
          addParent
            ? fmt(t.addUnder ?? "Add under \u201c{name}\u201d", { name: addParent.title })
            : (addType === "folder" ? (t.addFolderTitle ?? "Add folder") : (t.addPageTitle ?? "Add page"))
        }
        footer={
          <>
            <Button variant="ghost" onClick={() => setAddOpen(false)}>
              {t.cancel ?? "Cancel"}
            </Button>
            <Button onClick={submitAdd} disabled={busy === "add" || !addTitle.trim()}>
              {busy === "add" ? <Spinner /> : <Plus size={14} weight="bold" />}
              {t.create ?? "Create"}
            </Button>
          </>
        }
      >
        <div className="grid gap-4">
          <div>
            <Label>{t.titleLabel ?? "Title"}</Label>
            <Input
              value={addTitle}
              onChange={(e) => setAddTitle(e.target.value)}
              placeholder={t.titlePlaceholder ?? "e.g. Architecture overview"}
              autoFocus
            />
          </div>
        </div>
      </Modal>

      {/* Rename modal */}
      <Modal
        open={!!renameNode}
        onClose={() => setRenameNode(null)}
        title={t.renameTitle ?? "Rename node"}
        footer={
          <>
            <Button variant="ghost" onClick={() => setRenameNode(null)}>
              {t.cancel ?? "Cancel"}
            </Button>
            <Button
              onClick={submitRename}
              disabled={busy === "rename" || !renameTitle.trim()}
            >
              {busy === "rename" ? <Spinner /> : <PencilSimple size={14} weight="regular" />}
              {t.save ?? "Save"}
            </Button>
          </>
        }
      >
        <Label>{t.titleLabel ?? "Title"}</Label>
        <Input
          value={renameTitle}
          onChange={(e) => setRenameTitle(e.target.value)}
          autoFocus
        />
      </Modal>
    </div>
  );
}

export default KnowledgeTree;
