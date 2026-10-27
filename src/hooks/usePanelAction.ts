"use client";

import { useCallback, useState } from "react";
import { useNotifications } from "@/contexts/NotificationContext";

/**
 * Shared busy-guard + error toast for panel mutations (users / schedules):
 * `run(fn)` ignores re-entries while busy, reports failures via the `fail`
 * toast, and resolves to true only when fn completed — callers gate
 * follow-up UI (e.g. clearing an input field) on that success.
 */
export function usePanelAction(title: string) {
  const { notify } = useNotifications();
  const [busy, setBusy] = useState(false);

  const fail = useCallback(
    (e: unknown) =>
      notify({
        tone: "error",
        title,
        message: e instanceof Error ? e.message : "",
      }),
    [notify, title],
  );

  const run = useCallback(
    async (fn: () => Promise<void>): Promise<boolean> => {
      if (busy) return false;
      setBusy(true);
      try {
        await fn();
        return true;
      } catch (e) {
        fail(e);
        return false;
      } finally {
        setBusy(false);
      }
    },
    [busy, fail],
  );

  return { busy, fail, run };
}
