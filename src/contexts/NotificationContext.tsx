"use client";

/**
 * Global in-session notification system.
 *
 * `notify()` pushes an item that appears as a transient toast (top-right) and
 * is retained in the notification tray until manually dismissed or cleared.
 * The toast auto-hides after a tone-based delay (info/success ~10s,
 * warning/error ~15s); the tray entry persists regardless.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

export type NotificationTone = "info" | "success" | "error" | "warning";

export interface AppNotification {
  id: string;
  tone: NotificationTone;
  title: string;
  message?: string;
  createdAt: number;
  seen: boolean;
  /** When true the floating toast is hidden, but the tray entry remains. */
  hidden: boolean;
}

interface NotifyInput {
  tone?: NotificationTone;
  title: string;
  message?: string;
}

interface NotificationContextValue {
  notifications: AppNotification[];
  unreadCount: number;
  notify: (input: NotifyInput) => string;
  dismiss: (id: string) => void;
  clearAll: () => void;
  markAllSeen: () => void;
}

const NotificationContext = createContext<NotificationContextValue | undefined>(
  undefined,
);

const TOAST_DURATION: Record<NotificationTone, number> = {
  info: 10_000,
  success: 10_000,
  warning: 15_000,
  error: 15_000,
};

let idCounter = 0;
function makeId(): string {
  idCounter += 1;
  return `ntf_${Date.now().toString(36)}_${idCounter}`;
}

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const markHidden = useCallback((id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, hidden: true } : n)),
    );
    timersRef.current.delete(id);
  }, []);

  const notify = useCallback(
    (input: NotifyInput): string => {
      const tone = input.tone ?? "info";
      const id = makeId();
      const item: AppNotification = {
        id,
        tone,
        title: input.title,
        message: input.message,
        createdAt: Date.now(),
        seen: false,
        hidden: false,
      };
      setNotifications((prev) => [item, ...prev]);
      const timer = setTimeout(() => markHidden(id), TOAST_DURATION[tone]);
      timersRef.current.set(id, timer);
      return id;
    },
    [markHidden],
  );

  const dismiss = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const clearAll = useCallback(() => {
    timersRef.current.forEach((t) => clearTimeout(t));
    timersRef.current.clear();
    setNotifications([]);
  }, []);

  const markAllSeen = useCallback(() => {
    setNotifications((prev) =>
      prev.map((n) => (n.seen ? n : { ...n, seen: true })),
    );
  }, []);

  // Clean up pending toast timers on unmount.
  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((t) => clearTimeout(t));
      timers.clear();
    };
  }, []);

  const unreadCount = useMemo(
    () => notifications.reduce((acc, n) => (n.seen ? acc : acc + 1), 0),
    [notifications],
  );

  const value = useMemo<NotificationContextValue>(
    () => ({
      notifications,
      unreadCount,
      notify,
      dismiss,
      clearAll,
      markAllSeen,
    }),
    [notifications, unreadCount, notify, dismiss, clearAll, markAllSeen],
  );

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotifications(): NotificationContextValue {
  const ctx = useContext(NotificationContext);
  if (ctx === undefined) {
    throw new Error("useNotifications must be used within a NotificationProvider");
  }
  return ctx;
}
