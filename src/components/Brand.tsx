"use client";

import Link from "next/link";
import { cn } from "@/components/ui";

/**
 * Productarium brand mark: the layered-tile logo (currentColor) + wordmark.
 * Used in the TopBar `left` slot across all pages.
 */
export function BrandLogo({ className }: { className?: string }) {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 28 28"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("text-ink", className)}
      role="img"
      aria-label="Productarium"
    >
      <rect x="3" y="3" width="22" height="22" rx="6" fill="none" stroke="currentColor" strokeWidth="1.6" opacity="0.35" />
      <rect x="6.5" y="6.5" width="15" height="15" rx="4" fill="none" stroke="currentColor" strokeWidth="1.6" opacity="0.6" />
      <path d="M10 9.5h5.2a3.3 3.3 0 0 1 0 6.6H12.4V19H10V9.5Z" fill="currentColor" />
      <circle cx="18.2" cy="18.2" r="1.6" fill="currentColor" opacity="0.55" />
    </svg>
  );
}

export function Brand({
  href = "/",
  className,
}: {
  href?: string;
  className?: string;
}) {
  return (
    <Link href={href} className={cn("flex items-center gap-2", className)}>
      <BrandLogo />
      <span className="font-editorial text-base tracking-tight text-ink">
        Productarium
      </span>
    </Link>
  );
}

export default Brand;
