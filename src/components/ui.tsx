"use client";

/**
 * minimalist-ui building blocks (Notion / Linear editorial).
 * Warm monochrome, 1px #EAEAEA dividers, solid ink buttons, desaturated
 * pastel tags, quiet IntersectionObserver reveals. No gradients / heavy
 * shadows. Icons are Phosphor (@phosphor-icons/react).
 */

import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { CircleNotch } from "@phosphor-icons/react";
import type { TagTone } from "@/lib/types";

/* ------------------------------------------------------------------ */
/* Class helpers                                                       */
/* ------------------------------------------------------------------ */

export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

const TAG_TONES: Record<TagTone, string> = {
  blue: "bg-tag-blue-bg text-tag-blue-fg",
  green: "bg-tag-green-bg text-tag-green-fg",
  yellow: "bg-tag-yellow-bg text-tag-yellow-fg",
  red: "bg-tag-red-bg text-tag-red-fg",
  neutral: "bg-tag-neutral-bg text-tag-neutral-fg",
};

/* ------------------------------------------------------------------ */
/* Tag — pill, uppercase, pastel                                       */
/* ------------------------------------------------------------------ */

export function Tag({
  tone = "neutral",
  children,
  className,
}: {
  tone?: TagTone;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium uppercase tracking-wide whitespace-nowrap",
        TAG_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Button — solid ink, no shadow, active scale(0.98)                   */
/* ------------------------------------------------------------------ */

type ButtonVariant = "primary" | "ghost" | "danger" | "subtle";
type ButtonSize = "sm" | "md";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--button-bg)] text-[var(--button-fg)] hover:bg-[var(--button-hover)] border border-transparent",
  ghost: "bg-transparent text-ink border border-divider hover:bg-surface-2",
  danger:
    "bg-transparent text-tag-red-fg border border-divider hover:bg-tag-red-bg",
  subtle: "bg-surface-2 text-ink border border-transparent hover:bg-divider",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: "h-9 px-3 text-sm rounded-md gap-1.5",
  md: "h-11 px-4 text-[15px] rounded-md gap-2",
};

export function Button({
  variant = "primary",
  size = "md",
  className,
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center font-medium transition-[background-color,transform,border-color] duration-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50",
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function IconButton({
  className,
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cn(
        "inline-flex h-9 w-9 items-center justify-center rounded-md border border-divider bg-surface text-ink transition-colors duration-200 hover:bg-surface-2 active:scale-[0.98] disabled:opacity-50",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Card — 1px border, radius 12px, quiet hover lift                    */
/* ------------------------------------------------------------------ */

export function Card({
  className,
  children,
  hover = false,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { hover?: boolean }) {
  return (
    <div
      className={cn(
        "rounded-xl border border-divider bg-surface",
        hover && "card-hover",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Section header — editorial serif                                    */
/* ------------------------------------------------------------------ */

export function SectionHeader({
  title,
  subtitle,
  action,
  className,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-4", className)}>
      <div className="min-w-0">
        <h2 className="font-editorial text-2xl tracking-tight text-ink">
          {title}
        </h2>
        {subtitle && (
          <p className="mt-1 text-[15px] text-muted">{subtitle}</p>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Reveal — IntersectionObserver translateY(12px)+opacity, 600ms       */
/* ------------------------------------------------------------------ */

export function Reveal({
  children,
  delayMs = 0,
  className,
  as: TagEl = "div",
}: {
  children: React.ReactNode;
  delayMs?: number;
  className?: string;
  as?: React.ElementType;
}) {
  const ref = useRef<HTMLElement | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setVisible(true);
            observer.disconnect();
          }
        }
      },
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <TagEl
      ref={ref}
      className={cn("reveal", visible && "is-visible", className)}
      style={{ "--reveal-delay": `${delayMs}ms` } as React.CSSProperties}
    >
      {children}
    </TagEl>
  );
}

/* ------------------------------------------------------------------ */
/* Form controls — 1px border, radius 6px, focus border ink            */
/* ------------------------------------------------------------------ */

export function Label({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <label
      className={cn(
        "mb-1.5 block text-[13px] font-medium uppercase tracking-wide text-muted",
        className,
      )}
    >
      {children}
    </label>
  );
}

const FIELD_BASE =
  "w-full rounded-md border border-divider bg-surface px-3 py-2.5 text-base text-ink placeholder:text-[#a8a6a2] transition-colors duration-200 focus:border-ink focus:outline-none";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function Input({ className, ...props }, ref) {
  return <input ref={ref} className={cn(FIELD_BASE, className)} {...props} />;
});

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className, ...props }, ref) {
  return (
    <textarea
      ref={ref}
      className={cn(FIELD_BASE, "font-mono leading-relaxed", className)}
      {...props}
    />
  );
});

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(function Select({ className, children, ...props }, ref) {
  return (
    <select
      ref={ref}
      className={cn(FIELD_BASE, "appearance-none pr-8", className)}
      {...props}
    >
      {children}
    </select>
  );
});

/* ------------------------------------------------------------------ */
/* Empty state + spinner                                               */
/* ------------------------------------------------------------------ */

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border border-dashed border-divider bg-surface px-6 py-16 text-center",
        className,
      )}
    >
      {icon && (
        <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-surface-2 text-muted">
          {icon}
        </div>
      )}
      <h3 className="font-editorial text-xl text-ink">{title}</h3>
      {description && (
        <p className="mt-2 max-w-sm text-base text-muted">{description}</p>
      )}
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <CircleNotch
      className={cn("animate-spin", className)}
      size={16}
      weight="bold"
      aria-hidden
    />
  );
}

/* ------------------------------------------------------------------ */
/* Top bar — minimal sticky header used across pages                   */
/* ------------------------------------------------------------------ */

export function TopBar({
  left,
  center,
  right,
}: {
  left: React.ReactNode;
  center?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-divider bg-canvas/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 items-center justify-between gap-4 px-6">
        <div className="flex items-center gap-2">{left}</div>
        {center && (
          <div className="hidden items-center gap-1 md:flex">{center}</div>
        )}
        {right && <div className="flex items-center gap-2">{right}</div>}
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* Banner — inline alert (info / success / error)                      */
/* ------------------------------------------------------------------ */

export type BannerTone = "info" | "success" | "error" | "warning";

const BANNER_TONES: Record<BannerTone, string> = {
  info: "border-tag-blue-bg bg-tag-blue-bg text-tag-blue-fg",
  success: "border-tag-green-bg bg-tag-green-bg text-tag-green-fg",
  error: "border-tag-red-bg bg-tag-red-bg text-tag-red-fg",
  warning: "border-tag-yellow-bg bg-tag-yellow-bg text-tag-yellow-fg",
};

export function Banner({
  tone = "info",
  children,
  className,
  onClose,
}: {
  tone?: BannerTone;
  children: React.ReactNode;
  className?: string;
  onClose?: () => void;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-md border px-4 py-3 text-[15px]",
        BANNER_TONES[tone],
        className,
      )}
      role={tone === "error" ? "alert" : "status"}
    >
      <div className="flex-1">{children}</div>
      {onClose && (
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded p-0.5 text-current opacity-60 transition-opacity hover:opacity-100"
          aria-label="Dismiss"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Modal — centered dialog with backdrop, ESC to close                 */
/* ------------------------------------------------------------------ */

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  size = "md",
}: {
  open: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: "sm" | "md" | "lg" | "xl";
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  // Render into document.body via a portal so the dialog escapes any ancestor
  // stacking context (e.g. the .reveal class used by <Reveal>, which keeps
  // `transform`/`will-change` even when visible and would otherwise trap a
  // `position: fixed` modal — causing sibling cards to overlap it).
  if (!open || typeof document === "undefined") return null;
  const widths = {
    sm: "max-w-md",
    md: "max-w-lg",
    lg: "max-w-2xl",
    xl: "max-w-4xl",
  };
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />
      <div
        role="dialog"
        aria-modal="true"
        className={cn(
          "relative my-auto flex max-h-[90vh] w-full flex-col rounded-xl border border-divider bg-surface shadow-[0_8px_40px_rgba(0,0,0,0.18)]",
          widths[size],
        )}
      >
        {title && (
          <div className="flex items-center justify-between border-b border-divider px-5 py-4">
            <h3 className="font-editorial text-lg tracking-tight text-ink">
              {title}
            </h3>
            <IconButton aria-label="Close" onClick={onClose}>
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </IconButton>
          </div>
        )}
        <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
        {footer && (
          <div className="flex items-center justify-end gap-2 border-t border-divider px-5 py-4">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

/* ------------------------------------------------------------------ */
/* Switch — toggle control                                             */
/* ------------------------------------------------------------------ */

export function Switch({
  checked,
  onChange,
  disabled,
  label,
  className,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  label?: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border border-divider transition-colors duration-200 disabled:opacity-50",
        checked ? "bg-ink" : "bg-surface-2",
        className,
      )}
    >
      <span
        className={cn(
          "inline-block h-3.5 w-3.5 transform rounded-full bg-surface shadow-sm transition-transform duration-200",
          checked ? "translate-x-[18px]" : "translate-x-[3px]",
        )}
      />
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Avatar — initials circle                                            */
/* ------------------------------------------------------------------ */

export function Avatar({
  name,
  className,
  size = 28,
}: {
  name: string;
  className?: string;
  size?: number;
}) {
  const initials = name
    .split(/[\s_.-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]?.toUpperCase() ?? "")
    .join("");
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center rounded-full bg-surface-2 text-ink",
        className,
      )}
      style={{ width: size, height: size, fontSize: Math.max(10, size * 0.38) }}
      aria-hidden
    >
      {initials || "?"}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Divider + Skeleton                                                  */
/* ------------------------------------------------------------------ */

export function Divider({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-divider", className)} />;
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-surface-2", className)}
      aria-hidden
    />
  );
}
