"use client";

import { useState } from "react";
import { SealCheck, SealQuestion, Spinner } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { Button, Tag, cn } from "@/components/ui";
import { useAuth } from "@/contexts/AuthContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { useLanguage } from "@/contexts/LanguageContext";

/** Read-only "Verified" pill shown when an entity is verified. */
export function VerifiedBadge({
  verified,
  verifiedBy,
  className,
}: {
  verified?: boolean;
  verifiedBy?: string | null;
  className?: string;
}) {
  const { messages, fmt } = useLanguage();
  const t = messages?.verified ?? {};
  if (!verified) return null;
  return (
    <Tag tone="green" className={cn("gap-1", className)}>
      <SealCheck size={12} weight="fill" />
      {verifiedBy ? fmt(t.verifiedBy ?? "Verified · {name}", { name: verifiedBy }) : (t.verified ?? "Verified")}
    </Tag>
  );
}

/**
 * Verified toggle button. POSTs to `verifyUrl` to flip the flag. Only
 * owner/admin may verify — `canVerify` is derived from the current user role
 * (admin always; owner when `ownerId` matches). When disabled, renders the
 * badge only (no button).
 */
export function VerifiedButton({
  verified,
  verifyUrl,
  ownerId,
  onVerified,
  size = "sm",
}: {
  verified?: boolean;
  verifyUrl: string;
  ownerId?: string | null;
  onVerified?: (next: { verified: boolean; verified_by?: string | null }) => void;
  size?: "sm" | "md";
}) {
  const { user } = useAuth();
  const { notify } = useNotifications();
  const router = useRouter();
  const { messages } = useLanguage();
  const t = messages?.verified ?? {};
  const [busy, setBusy] = useState(false);

  const isAdmin = user?.role === "admin";
  const isOwner = !!user && !!ownerId && user.id === ownerId;
  const canVerify = isAdmin || isOwner;

  if (!canVerify) {
    return verified ? <VerifiedBadge verified={verified} /> : null;
  }

  const toggle = async () => {
    setBusy(true);
    try {
      const res = await fetch(verifyUrl, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      });
      if (res.status === 401) {
        router.replace("/login");
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Verify failed (${res.status})`);
      }
      const data = await res.json().catch(() => ({}));
      onVerified?.({
        verified: data.verified ?? !verified,
        verified_by: data.verified_by ?? null,
      });
    } catch (e) {
      notify({
        tone: "error",
        title: t.failedTitle ?? "Verify failed",
        message: e instanceof Error ? e.message : (t.failedMessage ?? "Verify failed"),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <Button
        type="button"
        size={size}
        variant={verified ? "subtle" : "ghost"}
        onClick={toggle}
        disabled={busy}
        className={verified ? "text-tag-green-fg" : undefined}
      >
        {busy ? (
          <Spinner />
        ) : verified ? (
          <SealCheck size={14} weight="fill" />
        ) : (
          <SealQuestion size={14} weight="regular" />
        )}
        {verified ? (t.verified ?? "Verified") : (t.markVerified ?? "Mark verified")}
      </Button>
    </div>
  );
}

export default VerifiedBadge;
