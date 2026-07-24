"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ArrowRight,
  Fingerprint,
  Lock,
  ShieldStar,
  Spinner,
  UserCircle,
} from "@phosphor-icons/react";
import { Brand } from "@/components/Brand";
import { Banner, Button, Card, Input, Label, TopBar } from "@/components/ui";
import { useAuth } from "@/contexts/AuthContext";
import { useLanguage } from "@/contexts/LanguageContext";
import type { SetupStatus } from "@/lib/types";

/**
 * Productarium sign-in (contract J) + first-run admin setup.
 *
 * On mount we probe GET /api/auth/setup-status:
 * - setup_required=true  -> show a "Create admin" form (POST /api/auth/setup)
 *   so the very first visitor can bootstrap the platform with an admin account.
 * - setup_required=false -> show the normal sign-in form (local + Keycloak).
 *
 * A "Forgot password?" link points to /reset-password, where a reset token
 * (issued by the admin at user creation) can be used to set a new password.
 */
function AuthForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { user, login, refresh, initialized } = useAuth();
  const { messages } = useLanguage();
  const t = messages?.auth ?? {};

  const [setup, setSetup] = useState<SetupStatus | null>(null);
  const [setupLoading, setSetupLoading] = useState(true);

  // Login form state
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [keycloakState, setKeycloakState] = useState<
    "idle" | "checking" | "configured" | "unconfigured"
  >("idle");

  // Setup form state
  const [suUsername, setSuUsername] = useState("");
  const [suPassword, setSuPassword] = useState("");
  const [suConfirm, setSuConfirm] = useState("");
  const [suEmail, setSuEmail] = useState("");
  const [suSubmitting, setSuSubmitting] = useState(false);

  const next = params.get("next") || "/";

  // Probe setup-status once.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/auth/setup-status", {
          credentials: "include",
          cache: "no-store",
        });
        if (!res.ok) {
          if (!cancelled) setSetup({ setup_required: false, auth_provider: "local" });
          return;
        }
        const data = (await res.json()) as SetupStatus;
        if (!cancelled) setSetup(data);
      } catch {
        if (!cancelled) setSetup({ setup_required: false, auth_provider: "local" });
      } finally {
        if (!cancelled) setSetupLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Redirect already-authenticated users away from /login.
  useEffect(() => {
    if (initialized && user) {
      router.replace(next);
    }
  }, [initialized, user, next, router]);

  // Best-effort Keycloak availability probe (only when not in setup mode).
  useEffect(() => {
    if (setupLoading || setup?.setup_required) return;
    let cancelled = false;
    (async () => {
      setKeycloakState("checking");
      try {
        const res = await fetch("/api/auth/keycloak/login", {
          credentials: "include",
          redirect: "manual",
        });
        if (cancelled) return;
        if (res.type === "opaqueredirect" || (res.status >= 200 && res.status < 400)) {
          setKeycloakState("configured");
        } else if (res.status === 501) {
          setKeycloakState("unconfigured");
        } else {
          setKeycloakState("configured");
        }
      } catch {
        if (!cancelled) setKeycloakState("unconfigured");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setupLoading, setup?.setup_required]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await login(username.trim(), password);
      router.replace(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : (t.loginFailed ?? "Login failed"));
    } finally {
      setSubmitting(false);
    }
  };

  const handleSetup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!suUsername.trim() || !suPassword || suSubmitting) return;
    if (suPassword !== suConfirm) {
      setError(t.passwordsNoMatch ?? "Passwords do not match.");
      return;
    }
    setSuSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/setup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          username: suUsername.trim(),
          password: suPassword,
          email: suEmail.trim() || undefined,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Setup failed (${res.status})`);
      }
      await refresh();
      router.replace(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : (t.setupFailed ?? "Setup failed"));
    } finally {
      setSuSubmitting(false);
    }
  };

  const handleKeycloak = () => {
    window.location.href = "/api/auth/keycloak/login";
  };

  if (setupLoading) {
    return (
      <div className="min-h-screen bg-canvas text-ink">
        <TopBar left={<Brand />} />
        <main className="mx-auto flex max-w-md flex-col px-6 py-20">
          <div className="flex items-center gap-2 text-sm text-muted">
            <Spinner /> {t.loading ?? "Loading…"}
          </div>
        </main>
      </div>
    );
  }

  if (setup?.setup_required) {
    return (
      <div className="min-h-screen bg-canvas text-ink">
        <TopBar left={<Brand />} />
        <main className="mx-auto flex max-w-md flex-col px-6 py-20">
          <div className="mb-8 text-center">
            <h1 className="font-editorial text-2xl tracking-tight text-ink">
              {t.createAdminTitle ?? "Create the admin account"}
            </h1>
            <p className="mt-2 text-base text-muted">
              {t.createAdminDesc ?? ""}
            </p>
          </div>
          <Card className="p-6">
            <form onSubmit={handleSetup} className="grid gap-4">
              <div>
                <Label>{t.adminUsername ?? "Admin username"}</Label>
                <Input
                  value={suUsername}
                  onChange={(e) => setSuUsername(e.target.value)}
                  placeholder={t.adminUsernamePlaceholder ?? "admin"}
                  autoComplete="username"
                  required
                  autoFocus
                />
              </div>
              <div>
                <Label>{t.emailOptional ?? "Email (optional)"}</Label>
                <Input
                  type="email"
                  value={suEmail}
                  onChange={(e) => setSuEmail(e.target.value)}
                  placeholder={t.emailPlaceholder ?? "admin@example.com"}
                  autoComplete="email"
                />
              </div>
              <div>
                <Label>{t.password ?? "Password"}</Label>
                <Input
                  type="password"
                  value={suPassword}
                  onChange={(e) => setSuPassword(e.target.value)}
                  placeholder={t.passwordPlaceholder ?? "password"}
                  autoComplete="new-password"
                  required
                />
              </div>
              <div>
                <Label>{t.confirmPassword ?? "Confirm password"}</Label>
                <Input
                  type="password"
                  value={suConfirm}
                  onChange={(e) => setSuConfirm(e.target.value)}
                  placeholder={t.setupRepeatPlaceholder ?? "repeat password"}
                  autoComplete="new-password"
                  required
                />
              </div>
              {error && <Banner tone="error">{error}</Banner>}
              <Button
                type="submit"
                disabled={suSubmitting || !suUsername.trim() || !suPassword || !suConfirm}
              >
                {suSubmitting ? <Spinner /> : <ShieldStar size={16} weight="fill" />}
                {suSubmitting ? (t.creatingAdmin ?? "Creating admin…") : (t.createAdminAction ?? "Create admin & sign in")}
              </Button>
            </form>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <TopBar left={<Brand />} />
      <main className="mx-auto flex max-w-md flex-col px-6 py-20">
        <div className="mb-8 text-center">
          <h1 className="font-editorial text-2xl tracking-tight text-ink">
            {t.signInTitle ?? "Sign in to Productarium"}
          </h1>
          <p className="mt-2 text-base text-muted">
            {t.signInDesc ?? ""}
          </p>
        </div>

        <Card className="p-6">
          <form onSubmit={handleSubmit} className="grid gap-4">
            <div>
              <Label>{t.username ?? "Username"}</Label>
              <Input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder={t.usernamePlaceholder ?? "username"}
                autoComplete="username"
                required
                autoFocus
              />
            </div>
            <div>
              <Label>{t.password ?? "Password"}</Label>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={t.passwordPlaceholder ?? "password"}
                autoComplete="current-password"
                required
              />
            </div>

            {error && <Banner tone="error">{error}</Banner>}

            <Button type="submit" disabled={submitting || !username.trim() || !password}>
              {submitting ? <Spinner /> : <Lock size={16} weight="fill" />}
              {submitting ? (t.signingIn ?? "Signing in…") : (t.signInAction ?? "Sign in")}
            </Button>

            <div className="text-right">
              <Link
                href="/reset-password"
                className="text-xs text-muted underline underline-offset-2 hover:text-ink"
              >
                {t.forgotPassword ?? "Forgot password / have a reset token?"}
              </Link>
            </div>
          </form>

          {keycloakState !== "unconfigured" && (
            <>
              <div className="my-5 flex items-center gap-3 text-xs text-muted">
                <div className="h-px flex-1 bg-divider" />
                {t.or ?? "or"}
                <div className="h-px flex-1 bg-divider" />
              </div>
              <Button
                type="button"
                variant="ghost"
                className="w-full"
                onClick={handleKeycloak}
                disabled={keycloakState === "checking"}
              >
                {keycloakState === "checking" ? (
                  <Spinner />
                ) : (
                  <Fingerprint size={16} weight="regular" />
                )}
                {t.continueWithKeycloak ?? "Continue with Keycloak"}
                <ArrowRight size={14} weight="bold" />
              </Button>
            </>
          )}
        </Card>

        <p className="mt-6 flex items-center justify-center gap-1.5 text-xs text-muted">
          <UserCircle size={14} weight="regular" />
          {t.sessionNote ?? ""}
        </p>
      </main>
    </div>
  );
}

// useSearchParams must be inside a Suspense boundary for Next 15 builds.
export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <AuthForm />
    </Suspense>
  );
}
