"use client";

import { useEffect, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Spinner } from "@/components/ui";
import { useAuth } from "@/contexts/AuthContext";

/**
 * Client-side route guard. Redirects to /login when the auth probe finishes
 * with no user. When `requireAdmin` is set, non-admins are bounced to /.
 *
 * With AUTH_PROVIDER=none the backend returns a `system` admin user, so the
 * guard lets the viewer through (no redirect).
 *
 * Note: this is a UX guard only — the backend enforces real auth on every
 * endpoint via the session cookie / API token. Never rely on this for security.
 */
export function AuthGuard({
  children,
  requireAdmin = false,
}: {
  children: ReactNode;
  requireAdmin?: boolean;
}) {
  const { user, initialized } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!initialized) return;
    if (!user) {
      router.replace("/login");
      return;
    }
    if (requireAdmin && user.role !== "admin") {
      router.replace("/");
    }
  }, [initialized, user, requireAdmin, router]);

  if (!initialized || !user || (requireAdmin && user.role !== "admin")) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center text-muted">
        <Spinner />
      </div>
    );
  }

  return <>{children}</>;
}

export default AuthGuard;
