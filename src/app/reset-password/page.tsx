"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Key, Spinner } from "@phosphor-icons/react";
import { Brand } from "@/components/Brand";
import { useNotifications } from "@/contexts/NotificationContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { Button, Card, Input, Label, TopBar } from "@/components/ui";

/**
 * Password reset by token (contract J).
 *
 * A reset token is generated when an admin creates a local user (or issues a
 * fresh one from the admin panel). The token can be pasted here (or pre-filled
 * via ?token=...) together with a new password to POST /api/auth/reset-password.
 * On success the user is redirected to /login.
 */
function ResetForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { messages } = useLanguage();
  const t = messages?.auth ?? {};

  const [token, setToken] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [mismatch, setMismatch] = useState(false);
  const [done, setDone] = useState(false);
  const { notify } = useNotifications();

  // Pre-fill the token from ?token=...
  useEffect(() => {
    const t = params.get("token");
    if (t) setToken(t);
  }, [params]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token || !password || submitting) return;
    if (password !== confirm) {
      setMismatch(true);
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ token, new_password: password }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Reset failed (${res.status})`);
      }
      setDone(true);
      notify({ tone: "success", title: t.resetDone ?? "Password reset" });
      setTimeout(() => router.replace("/login"), 1500);
    } catch (err) {
      notify({
        tone: "error",
        title: t.resetFailedTitle ?? "Reset failed",
        message: err instanceof Error ? err.message : (t.resetFailed ?? "Reset failed"),
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <TopBar left={<Brand />} />
      <main className="mx-auto flex max-w-md flex-col px-6 py-20">
        <Link
          href="/login"
          className="mb-6 inline-flex items-center gap-1 text-xs font-medium text-muted hover:text-ink"
        >
          <ArrowLeft size={14} weight="bold" /> {t.backToSignIn ?? "Back to sign in"}
        </Link>

        <div className="mb-8 text-center">
          <h1 className="font-editorial text-2xl tracking-tight text-ink">
            {t.resetTitle ?? "Reset your password"}
          </h1>
          <p className="mt-2 text-sm text-muted">
            {t.resetDesc ?? ""}
          </p>
        </div>

        <Card className="p-6">
          {done ? (
            <p className="text-sm text-tag-green-fg">
              {t.resetSuccessTitle ?? "Password reset. Redirecting to sign in…"}
            </p>
          ) : (
            <form onSubmit={handleSubmit} className="grid gap-4">
              <div>
                <Label>{t.resetToken ?? "Reset token"}</Label>
                <Input
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder={t.resetTokenPlaceholder ?? "paste the reset token"}
                  required
                  autoFocus={!token}
                  className="font-mono text-xs"
                />
              </div>
              <div>
                <Label>{t.newPassword ?? "New password"}</Label>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={t.newPasswordPlaceholder ?? "new password"}
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
                  placeholder={t.repeatPasswordPlaceholder ?? "repeat new password"}
                  autoComplete="new-password"
                  required
                />
              </div>
              {mismatch && (
                <p className="text-sm text-tag-red-fg">{t.passwordsNoMatch ?? "Passwords do not match."}</p>
              )}
              <Button type="submit" disabled={submitting || !token || !password || !confirm}>
                {submitting ? <Spinner /> : <Key size={16} weight="fill" />}
                {submitting ? (t.resetting ?? "Resetting…") : (t.resetAction ?? "Reset password")}
              </Button>
            </form>
          )}
        </Card>
      </main>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={null}>
      <ResetForm />
    </Suspense>
  );
}
