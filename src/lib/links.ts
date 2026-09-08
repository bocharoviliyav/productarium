/**
 * External link safety helpers (P0-4).
 *
 * Link targets (`links.content`, generated docs, spec artifacts) are user- or
 * model-authored and therefore untrusted. A raw `javascript:…` / `data:…`
 * href rendered into an `<a>` executes on click, so every external anchor in
 * the app must pass its href through `safeExternalHref` and render plain text
 * when the scheme is not explicitly allowed.
 */

/** URL schemes that are safe to open from untrusted content. */
export const SAFE_HREF_SCHEMES: ReadonlySet<string> = new Set([
  "http:",
  "https:",
  "mailto:",
]);

/**
 * Validate an untrusted href against the safe-scheme allowlist.
 *
 * Returns the trimmed href when it parses as an absolute URL with an allowed
 * scheme — or is a pure same-page fragment (`#id`; can only scroll within
 * the current document, never navigate) — and `null` otherwise, including
 * scheme-relative (`//host`), protocol-relative,
 * `javascript:`/`data:`/`vbscript:` URLs and relative paths. Callers must
 * render a non-clickable element for `null`.
 */
export function safeExternalHref(
  href: string | null | undefined,
): string | null {
  if (!href) return null;
  const trimmed = href.trim();
  if (!trimmed) return null;
  // In-page fragment: inherently same-page navigation, always safe.
  if (trimmed.startsWith('#')) return trimmed;
  let parsed: URL;
  try {
    // No base URL: relative targets throw and are rejected outright.
    parsed = new URL(trimmed);
  } catch {
    return null;
  }
  return SAFE_HREF_SCHEMES.has(parsed.protocol) ? parsed.href : null;
}
