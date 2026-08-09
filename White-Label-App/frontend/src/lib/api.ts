/**
 * Minimal typed API client.
 *
 * Hand-written on purpose, and temporary: from Phase 1 this file is replaced by
 * a TypeScript client generated from the backend's OpenAPI schema, so the
 * request/response types can never drift from the Python models. What survives
 * the swap is the error handling below — one place that understands the API's
 * error envelope.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** The envelope every backend error uses. See backend/app/core/errors.py. */
type ErrorEnvelope = {
  error: {
    code: string;
    message: string;
    details: unknown;
    request_id: string | null;
  };
};

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;
  /** Quote this in a bug report — it ties the failure to a backend log line. */
  readonly requestId: string | null;

  constructor(status: number, envelope: ErrorEnvelope["error"]) {
    super(envelope.message);
    this.name = "ApiError";
    this.status = status;
    this.code = envelope.code;
    this.details = envelope.details;
    this.requestId = envelope.request_id;
  }
}

type FetchOptions = RequestInit & {
  /**
   * Non-2xx statuses to treat as success and parse normally.
   *
   * Needed for readiness: `/health/ready` answers 503 with a *valid, useful*
   * body when a dependency is down, and we want to render it rather than throw.
   */
  acceptStatuses?: number[];
};

export async function apiFetch<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { acceptStatuses = [], ...init } = options;

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...init.headers },
      // Phase 1 auth uses httpOnly cookies, which requires this and an explicit
      // CORS origin allowlist on the backend (never "*").
      credentials: "include",
    });
  } catch (cause) {
    // fetch only rejects on network-level failure — the API being down, DNS,
    // or CORS. Surface that distinctly from an HTTP error status.
    throw new ApiError(0, {
      code: "network_error",
      message: `Cannot reach the API at ${API_BASE}. Is the backend running?`,
      details: String(cause),
      request_id: null,
    });
  }

  if (!response.ok && !acceptStatuses.includes(response.status)) {
    let envelope: ErrorEnvelope["error"];
    try {
      envelope = ((await response.json()) as ErrorEnvelope).error;
    } catch {
      // A proxy or crash can produce a non-JSON body; don't mask it with a
      // JSON parse error.
      envelope = {
        code: `http_${response.status}`,
        message: response.statusText || "Request failed",
        details: null,
        request_id: response.headers.get("X-Request-ID"),
      };
    }
    throw new ApiError(response.status, envelope);
  }

  return (await response.json()) as T;
}

// ── Endpoint types ────────────────────────────────────────────────────────────
// These are replaced by generated types once the OpenAPI client lands.

export type Meta = {
  product_name: string;
  version: string;
  environment: string;
};

export type DependencyStatus = {
  name: string;
  ok: boolean;
  latency_ms: number;
  detail: string | null;
};

export type Readiness = {
  status: "ready" | "degraded";
  dependencies: DependencyStatus[];
};

export const api = {
  meta: () => apiFetch<Meta>("/api/v1/meta"),
  readiness: () => apiFetch<Readiness>("/health/ready", { acceptStatuses: [503] }),
};
