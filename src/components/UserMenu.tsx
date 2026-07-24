"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { CaretDown, Gear, Key, SignOut, UserCircle } from "@phosphor-icons/react";
import { Avatar, Button, Input, Label, Modal, Spinner, cn, IconButton } from "@/components/ui";
import { useAuth } from "@/contexts/AuthContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { useLanguage } from "@/contexts/LanguageContext";

/**
 * TopBar user menu: avatar + username + role badge, with a dropdown for
 * /admin (admins only) and logout. When auth is disabled (AUTH_PROVIDER=none)
 * the backend returns a `system` admin user — the menu still renders so the
 * admin link is reachable.
 */
export function UserMenu() {
  const { user, logout, initialized } = useAuth();
  const router = useRouter();
  const { messages } = useLanguage();
  const t = messages?.userMenu ?? {};
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const [pwOpen, setPwOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  if (!initialized) {
    return <div className="h-8 w-8 animate-pulse rounded-full bg-surface-2" aria-hidden />;
  }

  if (!user) {
    return (
      <Link
        href="/login"
        className="inline-flex h-8 items-center gap-1.5 rounded-md border border-divider px-3 text-xs font-medium text-ink transition-colors hover:bg-surface-2"
      >
        <UserCircle size={14} weight="regular" />
        {t.signIn ?? "Sign in"}
      </Link>
    );
  }

  const isAdmin = user.role === "admin";

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex h-8 items-center gap-2 rounded-md border border-divider px-2 transition-colors hover:bg-surface-2"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <Avatar name={user.username} size={22} />
        <span className="hidden max-w-[120px] truncate text-xs font-medium text-ink sm:block">
          {user.username}
        </span>
        <span
          className={cn(
            "rounded-full px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
            isAdmin
              ? "bg-tag-blue-bg text-tag-blue-fg"
              : "bg-tag-neutral-bg text-tag-neutral-fg",
          )}
        >
          {user.role}
        </span>
        <CaretDown size={12} weight="regular" className="text-muted" />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-10 z-50 w-56 rounded-md border border-divider bg-surface py-1 shadow-[0_8px_40px_rgba(0,0,0,0.12)]"
        >
          <div className="border-b border-divider px-3 py-2">
            <p className="truncate text-sm font-medium text-ink">
              {user.username}
            </p>
            {user.email && (
              <p className="truncate text-xs text-muted">{user.email}</p>
            )}
            <p className="mt-1 text-[10px] uppercase tracking-wide text-muted">
              {user.provider} · {user.role}
            </p>
          </div>
          {isAdmin && (
            <Link
              href="/admin"
              role="menuitem"
              onClick={() => setOpen(false)}
              className="flex items-center gap-2 px-3 py-2 text-sm text-ink transition-colors hover:bg-surface-2"
            >
              <Gear size={14} weight="regular" />
              {t.adminPanel ?? "Admin panel"}
            </Link>
          )}
          {user.provider === "local" && (
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setOpen(false);
                setPwOpen(true);
              }}
              className="flex w-full items-center gap-2 px-3 py-2 text-sm text-ink transition-colors hover:bg-surface-2"
            >
              <Key size={14} weight="regular" />
              {t.changePassword ?? "Change password"}
            </button>
          )}
          <button
            type="button"
            role="menuitem"
            onClick={async () => {
              setOpen(false);
              await logout();
              router.push("/login");
            }}
            className="flex w-full items-center gap-2 px-3 py-2 text-sm text-ink transition-colors hover:bg-surface-2"
          >
            <SignOut size={14} weight="regular" />
            {t.signOut ?? "Sign out"}
          </button>
        </div>
      )}

      {pwOpen && <ChangePasswordModal onClose={() => setPwOpen(false)} />}
    </div>
  );
}

/* Change-password modal: POST /api/auth/change-password (old -> new).
   The backend resolves the current user from the session cookie, so no user
   id is needed here. */
function ChangePasswordModal({ onClose }: { onClose: () => void }) {
  const { notify } = useNotifications();
  const { messages } = useLanguage();
  const t = messages?.userMenu ?? {};
  const [oldPw, setOldPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirm, setConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [mismatch, setMismatch] = useState(false);
  const [done, setDone] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!oldPw || !newPw || submitting) return;
    if (newPw !== confirm) {
      setMismatch(true);
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/change-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ old_password: oldPw, new_password: newPw }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Change failed (${res.status})`);
      }
      setDone(true);
      notify({ tone: "success", title: t.changedTitle ?? "Password changed" });
      setTimeout(onClose, 1200);
    } catch (err) {
      notify({
        tone: "error",
        title: t.failedTitle ?? "Change failed",
        message: err instanceof Error ? err.message : (t.failedMessage ?? "Change failed"),
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal open onClose={onClose} title={t.changePasswordTitle ?? "Change password"} size="sm">
      {done ? (
        <p className="text-sm text-tag-green-fg">{t.passwordChanged ?? "Password changed."}</p>
      ) : (
        <form onSubmit={submit} className="grid gap-4">
          <div>
            <Label>{t.currentPassword ?? "Current password"}</Label>
            <Input
              type="password"
              value={oldPw}
              onChange={(e) => setOldPw(e.target.value)}
              autoComplete="current-password"
              required
              autoFocus
            />
          </div>
          <div>
            <Label>{t.newPassword ?? "New password"}</Label>
            <Input
              type="password"
              value={newPw}
              onChange={(e) => setNewPw(e.target.value)}
              autoComplete="new-password"
              required
            />
          </div>
          <div>
            <Label>{t.confirmNewPassword ?? "Confirm new password"}</Label>
            <Input
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              autoComplete="new-password"
              required
            />
          </div>
          {mismatch && (
            <p className="text-sm text-tag-red-fg">{t.passwordsNoMatch ?? "New passwords do not match."}</p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>
              {t.cancel ?? "Cancel"}
            </Button>
            <Button type="submit" disabled={submitting || !oldPw || !newPw || !confirm}>
              {submitting ? <Spinner /> : <Key size={16} weight="fill" />}
              {submitting ? (t.saving ?? "Saving…") : (t.changePassword ?? "Change password")}
            </Button>
          </div>
        </form>
      )}
    </Modal>
  );
}

/** Minimal sign-in button for places that just need a link (kept for reuse). */
export function SignInButton() {
  const { messages } = useLanguage();
  const t = messages?.userMenu ?? {};
  return (
    <Link href="/login">
      <IconButton aria-label={t.signIn ?? "Sign in"} title={t.signIn ?? "Sign in"}>
        <UserCircle size={16} weight="regular" />
      </IconButton>
    </Link>
  );
}

export default UserMenu;
