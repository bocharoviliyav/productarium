"use client";

/**
 * Segmented language toggle (EN | RU) for the TopBar. Mirrors the theme
 * toggle's footprint. Uses LanguageContext (persists to localStorage).
 */

import { useLanguage } from "@/contexts/LanguageContext";
import { cn } from "@/components/ui";

const LANGS: Array<{ code: string; label: string }> = [
  { code: "en", label: "EN" },
  { code: "ru", label: "RU" },
];

export default function LanguageToggle() {
  const { language, setLanguage, messages } = useLanguage();
  const t = messages?.language ?? {};

  return (
    <div
      role="group"
      aria-label={t.ariaLabel ?? "Language"}
      className="inline-flex h-9 items-center rounded-md border border-divider bg-surface p-0.5"
    >
      {LANGS.map((l) => {
        const active = language === l.code;
        const title = l.code === "en" ? (t.english ?? "English") : (t.russian ?? "Русский");
        return (
          <button
            key={l.code}
            type="button"
            onClick={() => setLanguage(l.code)}
            aria-pressed={active}
            title={title}
            className={cn(
              "inline-flex h-8 items-center rounded-[5px] px-2.5 text-xs font-medium transition-colors",
              active
                ? "bg-ink text-surface"
                : "text-muted hover:text-ink",
            )}
          >
            {l.label}
          </button>
        );
      })}
    </div>
  );
}
