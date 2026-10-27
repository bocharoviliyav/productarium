/**
 * Shared JSON fetch client (SPEC-003: «Failed to fetch» resilience).
 *
 * Every non-SSE API call goes through here to get:
 * - a per-request AbortController timeout;
 * - retries for idempotent GET only (network error / timeout / 5xx; 2 extra
 *   attempts, exponential backoff from 500 ms); 4xx never retries;
 * - caller-abort (component unmount / navigation) surfaced as ApiAbortError —
 *   callers swallow it silently, it is not an error state;
 * - ApiError.unavailable (network / 5xx) so GET callers keep stale data and
 *   show a quiet degraded indicator instead of an error banner.
 *
 * Terminal failures log exactly one `[api] METHOD url: reason` console line.
 */

const DEFAULT_TIMEOUT_MS = 15_000;
const GET_EXTRA_ATTEMPTS = 2;
const RETRY_BASE_DELAY_MS = 500;

/** Request failed at HTTP level (`status` set) or transport level (`status` null). */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** Backend unreachable or degraded — GET callers may keep stale data. */
  get unavailable(): boolean {
    return this.status === null || this.status >= 500;
  }
}

/** The caller-supplied AbortSignal fired (unmount/navigation) — not an error. */
export class ApiAbortError extends Error {
  constructor() {
    super("Request aborted by caller");
    this.name = "ApiAbortError";
  }
}

export interface ApiOptions {
  /** Component lifecycle signal; aborting it yields ApiAbortError. */
  signal?: AbortSignal;
  /** Per-request timeout in ms (default 15 s). */
  timeoutMs?: number;
}

/**
 * Central 401 handling for API callers: redirect to the login page with the
 * current route as `next`. Returns true when the error was a 401 (the caller
 * should stop); every other error is left to the caller.
 */
export function onApiError(
  e: unknown,
  router: { replace: (url: string) => void },
): boolean {
  if (!(e instanceof ApiError) || e.status !== 401) return false;
  if (typeof window !== "undefined") {
    const next = encodeURIComponent(window.location.pathname + window.location.search);
    router.replace(`/login?next=${next}`);
  }
  return true;
}

export function getJson<T>(url: string, opts?: ApiOptions): Promise<T> {
  return request<T>("GET", url, undefined, opts);
}

export function postJson<T>(
  url: string,
  body?: unknown,
  opts?: ApiOptions,
): Promise<T> {
  return request<T>("POST", url, body, opts);
}

export function putJson<T>(
  url: string,
  body?: unknown,
  opts?: ApiOptions,
): Promise<T> {
  return request<T>("PUT", url, body, opts);
}

export function patchJson<T>(
  url: string,
  body?: unknown,
  opts?: ApiOptions,
): Promise<T> {
  return request<T>("PATCH", url, body, opts);
}

/** DELETE; several product routes answer with the updated Product body. */
export function deleteJson<T = void>(
  url: string,
  opts?: ApiOptions,
): Promise<T> {
  return request<T>("DELETE", url, undefined, opts);
}

async function request<T>(
  method: string,
  url: string,
  body: unknown,
  opts: ApiOptions = {},
): Promise<T> {
  const attempts = method === "GET" ? 1 + GET_EXTRA_ATTEMPTS : 1;
  let lastError: unknown;
  try {
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      if (attempt > 0) {
        await delay(RETRY_BASE_DELAY_MS * 2 ** (attempt - 1));
        // Caller may have unmounted while we waited out the backoff.
        if (opts.signal?.aborted) throw new ApiAbortError();
      }
      try {
        return await readJson<T>(
          await fetchOnce(method, url, body, opts.signal, opts.timeoutMs),
        );
      } catch (e) {
        if (e instanceof ApiAbortError) throw e;
        lastError = e;
        if (!(e instanceof ApiError && e.unavailable)) throw e;
      }
    }
    throw lastError;
  } catch (e) {
    if (!(e instanceof ApiAbortError)) {
      console.warn(`[api] ${method} ${url} failed: ${describe(e)}`);
    }
    throw e;
  }
}

async function fetchOnce(
  method: string,
  url: string,
  body: unknown,
  signal: AbortSignal | undefined,
  timeoutMs: number | undefined,
): Promise<Response> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const onCallerAbort = () => controller.abort();
  // "abort" listeners do not fire retroactively — handle a pre-aborted signal.
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener("abort", onCallerAbort);

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      credentials: "include",
      cache: "no-store",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch {
    throw signal?.aborted
      ? new ApiAbortError()
      : new ApiError(timedOut ? "Request timed out" : "Network error", null);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onCallerAbort);
  }
  if (!response.ok) {
    throw new ApiError(
      (await readErrorDetail(response)) ?? `HTTP ${response.status}`,
      response.status,
    );
  }
  return response;
}

async function readJson<T>(response: Response): Promise<T> {
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  return (text ? (JSON.parse(text) as T) : undefined) as T;
}

/** Best-effort `{detail}` extraction from an error body (FastAPI contract). */
async function readErrorDetail(response: Response): Promise<string | null> {
  try {
    const data = (await response.json()) as { detail?: unknown };
    return typeof data?.detail === "string" ? data.detail : null;
  } catch {
    return null;
  }
}

function describe(e: unknown): string {
  return e instanceof ApiError
    ? e.status === null
      ? e.message
      : String(e.status)
    : "unknown";
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
