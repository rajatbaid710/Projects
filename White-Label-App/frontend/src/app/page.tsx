"use client";

/**
 * Phase 0 status page.
 *
 * Not decoration — this is the acceptance test for the foundation. Rendering it
 * successfully proves the whole chain works: browser -> CORS -> FastAPI ->
 * Postgres/Redis/Qdrant, plus versioned routing (/api/v1/meta), the unversioned
 * probe (/health/ready), the error envelope, and token-driven styling.
 *
 * It gets replaced by the real product shell in Phase 1.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, api, type Meta, type Readiness } from "@/lib/api";

type LoadState =
  | { kind: "loading" }
  | { kind: "ok"; meta: Meta; readiness: Readiness }
  | { kind: "error"; message: string; requestId: string | null };

export default function StatusPage() {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      // Both calls are independent; no reason to serialize them.
      const [meta, readiness] = await Promise.all([api.meta(), api.readiness()]);
      setState({ kind: "ok", meta, readiness });
    } catch (error) {
      const apiError = error instanceof ApiError ? error : null;
      setState({
        kind: "error",
        message: apiError?.message ?? String(error),
        requestId: apiError?.requestId ?? null,
      });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center gap-6 p-6">
      <header className="flex items-baseline justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          {state.kind === "ok" ? state.meta.product_name : "White Label App"}
        </h1>
        {state.kind === "ok" && (
          <span className="text-fg-muted font-mono text-xs">
            v{state.meta.version} · {state.meta.environment}
          </span>
        )}
      </header>

      <section className="border-border bg-surface-muted rounded-lg border p-5">
        <h2 className="text-fg-muted mb-4 text-xs font-medium uppercase tracking-wider">
          System status
        </h2>

        {state.kind === "loading" && <p className="text-fg-muted text-sm">Checking…</p>}

        {state.kind === "error" && (
          <div className="space-y-2">
            <p className="text-danger text-sm font-medium">{state.message}</p>
            <p className="text-fg-muted text-sm">
              Start the data services with{" "}
              <code className="font-mono text-xs">
                docker compose -f infra/docker-compose.yml up -d
              </code>
              , then the API with <code className="font-mono text-xs">uvicorn app.main:app --reload</code>.
            </p>
            {state.requestId && (
              <p className="text-fg-muted font-mono text-xs">request id: {state.requestId}</p>
            )}
          </div>
        )}

        {state.kind === "ok" && (
          <ul className="divide-border divide-y">
            {state.readiness.dependencies.map((dependency) => (
              <li key={dependency.name} className="flex items-center gap-3 py-2.5 text-sm">
                <span
                  aria-hidden
                  className={`size-2 shrink-0 rounded-full ${
                    dependency.ok ? "bg-success" : "bg-danger"
                  }`}
                />
                <span className="font-medium">{dependency.name}</span>
                <span className="text-fg-muted ml-auto font-mono text-xs">
                  {dependency.ok ? `${dependency.latency_ms} ms` : "unreachable"}
                </span>
                {/* Screen readers get the state that the colour dot conveys. */}
                <span className="sr-only">{dependency.ok ? "healthy" : "failing"}</span>
              </li>
            ))}
          </ul>
        )}

        {state.kind === "ok" &&
          state.readiness.dependencies
            .filter((dependency) => !dependency.ok && dependency.detail)
            .map((dependency) => (
              <p key={dependency.name} className="text-danger mt-3 font-mono text-xs">
                {dependency.name}: {dependency.detail}
              </p>
            ))}
      </section>

      <button
        type="button"
        onClick={() => void load()}
        disabled={state.kind === "loading"}
        className="bg-brand text-brand-contrast self-start rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50"
      >
        Refresh
      </button>
    </main>
  );
}
