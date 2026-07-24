"use client";

/**
 * Productarium auth context (contract J).
 *
 * Wraps the session-cookie auth flow: GET /api/auth/me to resolve the current
 * user, POST /api/auth/login for local login, POST /api/auth/logout to clear
 * the session. The session is an httpOnly cookie (`productarium_session`) set
 * by the backend, so we rely on `credentials: "include"` and never touch the
 * token in JS.
 *
 * When the backend runs with AUTH_PROVIDER=none, /api/auth/me returns a
 * bootstrap/system admin user — the UI then treats the viewer as signed in.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { User } from "@/lib/types";

type AuthError = string | null;

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  error: AuthError;
  /** Re-fetch the current user from /api/auth/me. */
  refresh: () => Promise<User | null>;
  /** Local username/password login. Returns the user on success. */
  login: (username: string, password: string) => Promise<User>;
  /** Clear the session cookie. */
  logout: () => Promise<void>;
  /** True once the initial /me probe has completed (regardless of result). */
  initialized: boolean;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<AuthError>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/me", {
        credentials: "include",
        cache: "no-store",
      });
      if (res.status === 401) {
        setUser(null);
        return null;
      }
      if (!res.ok) {
        // 404/500 etc. — treat as not authenticated but record the issue.
        setUser(null);
        setError(`Auth unavailable (${res.status})`);
        return null;
      }
      const data = (await res.json()) as User;
      setUser(data);
      return data;
    } catch (e) {
      // Network/backend down — degrade gracefully: no user, no hard error.
      setUser(null);
      setError(e instanceof Error ? e.message : "Auth check failed");
      return null;
    } finally {
      setLoading(false);
      setInitialized(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const login = useCallback(async (username: string, password: string) => {
    setError(null);
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      let detail = `Login failed (${res.status})`;
      try {
        const body = await res.json();
        detail = body?.detail || detail;
      } catch {
        /* ignore */
      }
      setError(detail);
      throw new Error(detail);
    }
    const data = (await res.json()) as User;
    setUser(data);
    return data;
  }, []);

  const logout = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch {
      /* ignore — clear local state regardless */
    } finally {
      setUser(null);
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ user, loading, error, refresh, login, logout, initialized }),
    [user, loading, error, refresh, login, logout, initialized],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (ctx === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
