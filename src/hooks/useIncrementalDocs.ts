"use client";

/**
 * useIncrementalDocs — single owner of the SPEC-010 detect + incremental
 * update flows, shared by the artifact viewer and SpecEditor.
 *
 * Backend contract off the entity base URL (`/api/products/{pid}/{segment}/{id}`):
 * - POST {base}/detect                      → {job_id}
 * - GET  {base}/detect/status?job_id=…      → {status, error?, result?}
 * - POST {base}/incremental                 → {job_id} (absent = nothing to do)
 * - GET  {base}/generate/status?job_id=…    → {status, error?, indexing_message?}
 *
 * Every request goes through src/lib/api.ts (GET retries + per-request
 * timeout ride the shared client), 401 redirects through onApiError, and
 * toasts stay silent once the host unmounts. `onApplied(outcome)` runs after
 * a run settles so the host refetches its product/version state.
 */

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/contexts/LanguageContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { getJson, onApiError, postJson } from "@/lib/api";
import type { DetectResult } from "@/lib/types";

const POLL_INTERVAL_MS = 2000;
const DETECT_WINDOW_MS = 10 * 60 * 1000;
const UPDATE_WINDOW_MS = 30 * 60 * 1000;

type DetectJobStatus = { status?: string; error?: string; result?: DetectResult | null };
type UpdateJobStatus = { status?: string; error?: string; indexing_message?: string };
type AppliedOutcome = "succeeded" | "cancelled" | "noop";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useIncrementalDocs({
  baseUrl,
  onApplied,
}: {
  /** Entity URL; empty while the entity is unresolved (handlers no-op). */
  baseUrl: string;
  onApplied: (outcome: AppliedOutcome) => void | Promise<void>;
}) {
  const router = useRouter();
  const { notify } = useNotifications();
  const { messages } = useLanguage();
  const t = messages?.artifact ?? {};

  // Silences poller toasts and state writes once the host unmounts.
  const abortRef = useRef(false);
  useEffect(() => {
    abortRef.current = false;
    return () => {
      abortRef.current = true;
    };
  }, []);

  const [detectChecking, setDetectChecking] = useState(false);
  const [detectResult, setDetectResult] = useState<DetectResult | null>(null);
  const [updateStarting, setUpdateStarting] = useState(false);
  const [updateJob, setUpdateJob] = useState<string | null>(null);

  const idle = !baseUrl || detectChecking || updateStarting || updateJob !== null;

  const reportError = (title: string, fallback: string, e: unknown) => {
    if (!onApiError(e, router) && !abortRef.current) {
      notify({
        tone: "error",
        title,
        message: e instanceof Error ? e.message : fallback,
      });
    }
  };

  // «Проверить изменения»: detect job → summary modal (changed=true) or an
  // info toast for up-to-date / unverifiable sources.
  const checkChanges = async () => {
    if (idle) return;
    setDetectChecking(true);
    try {
      const data = await postJson<{ job_id?: string | null }>(`${baseUrl}/detect`, {});
      const jobId = data.job_id ?? null;
      if (!jobId) throw new Error(t.detectFailed ?? "Failed to check changes.");
      const startedAt = Date.now();
      while (Date.now() - startedAt < DETECT_WINDOW_MS) {
        await delay(POLL_INTERVAL_MS);
        if (abortRef.current) return;
        const st = await getJson<DetectJobStatus>(
          `${baseUrl}/detect/status?job_id=${encodeURIComponent(jobId)}`,
        );
        if (st.status === "succeeded") {
          const result = st.result ?? null;
          if (result?.changed === true) {
            setDetectResult(result);
          } else {
            notify({
              tone: "info",
              title: t.detectTitle ?? "Change check",
              message: result?.reason || (t.detectNoChanges ?? "No source changes detected."),
            });
          }
          return;
        }
        if (st.status === "failed") {
          throw new Error(st.error || (t.detectFailed ?? "Failed to check changes."));
        }
      }
      throw new Error(t.detectFailed ?? "Failed to check changes.");
    } catch (e) {
      reportError(t.detectTitle ?? "Change check", t.detectFailed ?? "Failed to check changes.", e);
    } finally {
      if (!abortRef.current) setDetectChecking(false);
    }
  };

  // «Обновить только изменённое»: the worker re-detects at run start;
  // unchanged → success without a new version, changed → only the impacted
  // pages regenerate (empty impact → full regeneration).
  const applyUpdate = async () => {
    if (idle) return;
    setDetectResult(null);
    setUpdateStarting(true);
    try {
      const data = await postJson<{ job_id?: string | null }>(`${baseUrl}/incremental`, {});
      const jobId = data.job_id ?? null;
      if (!jobId) {
        await onApplied("noop");
        return;
      }
      notify({
        tone: "info",
        title: t.updateAll ?? "Update documentation",
        message: t.updateStarted ?? "Incremental update started…",
      });
      setUpdateJob(jobId);
      const startedAt = Date.now();
      while (Date.now() - startedAt < UPDATE_WINDOW_MS) {
        await delay(POLL_INTERVAL_MS);
        if (abortRef.current) return;
        const st = await getJson<UpdateJobStatus>(
          `${baseUrl}/generate/status?job_id=${encodeURIComponent(jobId)}`,
        );
        if (st.status === "succeeded") {
          notify({
            tone: "success",
            title: t.updateAll ?? "Update documentation",
            message: st.indexing_message || (t.updateDone ?? "Documentation updated."),
          });
          await onApplied("succeeded");
          return;
        }
        if (st.status === "failed") {
          throw new Error(st.error || st.indexing_message || (t.updateFailed ?? "Update failed."));
        }
        if (st.status === "cancelled") {
          notify({
            tone: "info",
            title: t.updateAll ?? "Update documentation",
            message: st.indexing_message || (t.updateCancelled ?? "Update cancelled."),
          });
          await onApplied("cancelled");
          return;
        }
      }
      throw new Error(t.updateFailed ?? "Update failed.");
    } catch (e) {
      reportError(t.updateAll ?? "Update documentation", t.updateFailed ?? "Update failed.", e);
    } finally {
      if (!abortRef.current) {
        setUpdateJob(null);
        setUpdateStarting(false);
      }
    }
  };

  return {
    detectChecking,
    detectResult,
    setDetectResult,
    updateStarting,
    updateJob,
    checkChanges,
    applyUpdate,
  };
}
