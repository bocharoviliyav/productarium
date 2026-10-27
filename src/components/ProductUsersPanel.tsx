"use client";

/**
 * «Пользователи продукта» (SPEC-001): per-product grant management, rendered
 * on the product page for users with manage rights (admin / owner / rw
 * grant). Owner and admins hold implicit access and never get rows. Admins
 * pick users by name (admin-only listing); managers type the raw user id.
 */

import { useCallback, useEffect, useState } from "react";
import { Plus, Trash } from "@phosphor-icons/react";
import { useAuth } from "@/contexts/AuthContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { usePanelAction } from "@/hooks/usePanelAction";
import {
  Button,
  Card,
  IconButton,
  Input,
  Label,
  SectionHeader,
  Select,
  Spinner,
} from "@/components/ui";
import { ApiError, deleteJson, getJson, putJson } from "@/lib/api";
import { type User, normalizeUsers } from "@/lib/types";

/** Row of GET /api/products/{id}/grants (username joined server-side). */
type GrantRow = {
  user_id: string;
  username?: string | null;
  level: "ro" | "rw";
};

export function ProductUsersPanel({ productId }: { productId: string }) {
  const { user } = useAuth();
  const { messages } = useLanguage();
  const t = messages?.product ?? {};
  const [rows, setRows] = useState<GrantRow[] | null>(null);
  const [candidates, setCandidates] = useState<User[] | null>(null);
  const [userId, setUserId] = useState("");
  const [level, setLevel] = useState<"ro" | "rw">("ro");
  const { busy, fail, run } = usePanelAction(t.usersTitle ?? "Product users");

  const load = useCallback(async () => {
    try {
      setRows(await getJson<GrantRow[]>(`/api/products/${productId}/grants`));
    } catch (e) {
      setRows([]);
      if (e instanceof ApiError && e.status === 403) return; // lost manage rights
      fail(e);
    }
  }, [productId, fail]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (user?.role !== "admin" || candidates !== null) return;
    void (async () => {
      try {
        setCandidates(normalizeUsers(await getJson("/api/admin/users")));
      } catch {
        setCandidates([]);
      }
    })();
  }, [user?.role, candidates]);

  const grant = (targetId: string, next: "ro" | "rw") =>
    run(async () => {
      await putJson(`/api/products/${productId}/grants`, {
        user_id: targetId,
        level: next,
      });
      await load();
    });

  const revoke = (targetId: string) =>
    run(async () => {
      await deleteJson(
        `/api/products/${productId}/grants?user_id=${encodeURIComponent(targetId)}`,
      );
      await load();
    });

  // Q-15: keep the typed id in the field unless the grant actually landed.
  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    const targetId = userId.trim();
    if (!targetId) return;
    if (await grant(targetId, level)) setUserId("");
  };

  return (
    <section>
      <SectionHeader title={t.usersTitle ?? "Product users"} subtitle={t.usersSubtitle ?? ""} />
      <Card className="mt-6 overflow-hidden">
        {/* Add grant */}
        <form onSubmit={add} className="grid gap-3 border-b border-divider p-4 md:grid-cols-[1fr_140px_auto]">
          <div>
            <Label>{t.usersUser ?? "User"}</Label>
            {candidates ? (
              <Select value={userId} onChange={(e) => setUserId(e.target.value)}>
                <option value="">{t.usersPickUser ?? "—"}</option>
                {candidates
                  .filter((u) => u.role !== "admin")
                  .map((u) => (
                    <option key={u.id} value={u.id}>
                      {u.username}
                    </option>
                  ))}
              </Select>
            ) : (
              <Input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder={t.usersUserIdPlaceholder ?? "User id"}
                className="font-mono"
                autoComplete="off"
              />
            )}
          </div>
          <div>
            <Label>{t.usersLevel ?? "Level"}</Label>
            <Select value={level} onChange={(e) => setLevel(e.target.value as "ro" | "rw")}>
              <option value="ro">{t.usersRo ?? "ro — read + docs"}</option>
              <option value="rw">{t.usersRw ?? "rw — manage"}</option>
            </Select>
          </div>
          <div className="flex items-end">
            <Button type="submit" size="sm" disabled={busy || !userId.trim()}>
              {busy ? <Spinner /> : <Plus size={14} weight="bold" />}
              {t.usersGrant ?? "Grant"}
            </Button>
          </div>
        </form>

        {/* Grants table */}
        {rows === null ? (
          <div className="flex items-center gap-2 p-4 text-sm text-muted">
            <Spinner /> {t.usersLoading ?? "Loading…"}
          </div>
        ) : rows.length === 0 ? (
          <p className="p-4 text-sm text-muted">{t.usersEmpty ?? "No grants yet."}</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="px-4 py-3 font-medium">{t.usersUser ?? "User"}</th>
                <th className="px-4 py-3 font-medium">{t.usersLevel ?? "Level"}</th>
                <th className="px-4 py-3 font-medium text-right">{t.usersRevoke ?? "Revoke"}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-divider">
              {rows.map((r) => (
                <tr key={r.user_id} className="hover:bg-surface-2">
                  <td className="px-4 py-3 font-medium text-ink">
                    {r.username || r.user_id}
                  </td>
                  <td className="px-4 py-3">
                    <Select
                      value={r.level}
                      onChange={(e) => void grant(r.user_id, e.target.value as "ro" | "rw")}
                      disabled={busy}
                      className="w-44 text-xs"
                    >
                      <option value="ro">{t.usersRo ?? "ro — read + docs"}</option>
                      <option value="rw">{t.usersRw ?? "rw — manage"}</option>
                    </Select>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <IconButton
                      type="button"
                      aria-label={t.usersRevoke ?? "Revoke"}
                      title={t.usersRevoke ?? "Revoke"}
                      onClick={() => void revoke(r.user_id)}
                      disabled={busy}
                    >
                      {busy ? <Spinner /> : <Trash size={14} weight="regular" />}
                    </IconButton>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </section>
  );
}
