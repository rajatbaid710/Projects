# White-Label App — Build Plan

A reusable foundation ("starter platform") that ships with authentication, user profiles, an admin module, role-based access control, security, **and an AI/agents layer** baked in — so any future product can start from a solid base instead of rebuilding plumbing. Targets **web + mobile** on a shared **Python** backend. The first product built on the base is a **document reader** (chat and Q&A over your documents), which exercises the AI, vector-search, and multi-agent capabilities end to end.

---

## Part 1 — High-Level Roadmap

### Vision

One backend API, two clients (web + mobile), and a shared design system that can be rebranded per product. The goal is not to build one app but to build the *thing you clone* to start the next ten apps. Every decision favors reusability, clear extension points, and sane defaults over cleverness.

### What "white-label" means here

The base must let a new product change identity and behavior without forking the code:

- **Branding via configuration** — name, logo, colors, fonts, and copy come from a theme/tenant config, not hardcoded values.
- **Feature flags** — turn modules on/off per deployment.
- **Multi-tenancy option** — either one deployment per client, or a single deployment serving many tenants (decide early; see spec).
- **Extension points** — clean seams where product-specific features plug in without touching core.

### Core feature scope

The base platform delivers: email/password and social login with secure sessions; a full user profile system (view, edit, avatar, account settings); role-based access control with roles and granular permissions; an admin module for managing users, roles, and settings; and a security layer covering password policies, token handling, rate limiting, and audit logging. On top of that sit the white-label essentials — theming, feature flags, and a settings framework — plus supporting infrastructure like notifications (email + push), file uploads, and observability.

The base also ships a reusable **AI layer**: a provider-agnostic LLM gateway, an embeddings + vector-search service, a document ingestion/RAG pipeline, and a multi-agent orchestration framework. Products opt into these via feature flags. The document reader is the first product to consume them.

### Delivery phases

**Phase 0 — Foundations (setup).** Monorepo, shared TypeScript config, CI/CD pipeline, database provisioning, environment/secrets management, and a "hello world" deploy of API + web + mobile. Nothing user-facing, but everything downstream depends on it.

**Phase 1 — Authentication & identity (MVP core).** Registration, email verification, login/logout, password reset, session/token management, and the user record. This is the spine — get it right and secure.

**Phase 2 — Profiles & RBAC.** User profile CRUD with avatar upload, account settings, plus the roles/permissions model and the middleware that enforces it across API and UI. Seed default roles (admin, user).

**Phase 3 — Admin module.** Admin dashboard to manage users (invite, suspend, delete, reset), assign roles, edit permissions, and view audit logs. This is where role-based access proves itself.

**Phase 4 — White-label layer.** Theme/branding configuration, feature flags, and the settings framework so a new product can rebrand and toggle modules without code changes.

**Phase 5 — AI foundation.** The reusable AI layer: an LLM gateway (provider-agnostic), an embeddings + vector-search service backed by the vector DB, and the document ingestion/RAG pipeline (upload → parse → chunk → embed → index → retrieve). Delivered as a base module, gated by a feature flag.

**Phase 6 — Multi-agent orchestration.** An agent framework on top of the AI foundation: agent definitions, tools, a planner/orchestrator, shared memory, and streaming of intermediate steps to the clients. Start with the agents the document reader needs, designed so new products can register their own.

**Phase 7 — First product: document reader.** The actual app built on the base — document upload/management UI, chat/Q&A over documents with cited sources, and the agents that summarize, extract, and answer. Proves the whole stack top to bottom.

**Phase 8 — Hardening & polish.** Security review, rate limiting, audit logging completeness, accessibility, error handling, notifications, documentation, and a "clone this to start a new app" guide.

Phases are sequential in dependency but the mobile and web clients are built in parallel against each API milestone. Phases 5–7 depend on the identity/RBAC core (1–3) but can begin once auth is stable.

### Milestones & rough sequencing

A realistic solo/small-team sequence: Foundations first, then a secure auth milestone, then profiles + RBAC as one milestone since they're tightly coupled, then the admin module, then the white-label layer, and finally a hardening pass before you'd call it a reusable base. Treat each phase as its own release with a demo and a checklist rather than fixing calendar dates up front.

### Priorities & principles

Security and auth are non-negotiable and come first — everything else builds on identity. RBAC should be designed once, early, and consistently enforced everywhere rather than bolted on. Favor boring, well-supported technology so the base stays maintainable. Document extension points as you build, because an undocumented starter kit is just someone else's codebase. Ship each phase behind feature flags so the base can be adopted incrementally.

---

## Part 2 — Technical Specification

### Recommended stack

A **Python backend** with TypeScript clients. Python is the right call here because the product is AI-heavy — the mature ecosystem for LLMs, embeddings, agents, and document parsing (LangChain/LlamaIndex, the official model SDKs, PDF/OCR tooling) lives in Python, and you want your agent and RAG code in the same language as that ecosystem.

| Layer | Recommendation | Why |
|---|---|---|
| Backend language | **Python 3.12+** | First-class AI/ML/agent ecosystem; async support |
| Backend framework | **FastAPI** | Async, type-hinted, auto OpenAPI docs, excellent for both CRUD and streaming AI endpoints |
| Data layer | **SQLAlchemy 2.0 + Alembic** | Mature ORM + migrations |
| Validation | **Pydantic v2** | Shared request/response models; pairs natively with FastAPI |
| Primary database | **PostgreSQL** | Relational integrity for users/roles/permissions; JSONB for settings/theme config |
| Vector database | **pgvector** (Postgres extension) to start; **Qdrant** if you outgrow it | pgvector keeps one database early and simplifies ops; Qdrant/Weaviate/Milvus are the graduation path for scale and advanced filtering |
| Task queue | **Celery** or **RQ** + Redis | Document ingestion/embedding are slow — run them as background jobs |
| LLM / agents | Provider SDKs behind a gateway; **LangGraph** or **LlamaIndex** for orchestration | Provider-agnostic; LangGraph gives explicit, debuggable multi-agent graphs |
| Web frontend | **Next.js (React, TypeScript)** | SSR/SEO, mature, huge ecosystem, easy streaming UI |
| Mobile | **React Native (Expo, TypeScript)** | Single codebase for iOS + Android; shares logic with web |
| Shared contract | **OpenAPI-generated TS client** from FastAPI | Keeps web + mobile in sync with the Python API automatically |
| Auth | JWT access + refresh tokens (in-house), or a managed provider (Clerk/Auth0/Cognito) as a pluggable option | Standard and portable, or offload security |
| State/data fetching | TanStack Query (web + RN) | Caching, retries, streaming |
| Styling/theming | Design tokens + Tailwind (web) / theme provider (RN) | Tokens drive white-label rebranding |
| Infra | Docker + docker-compose; managed Postgres + Redis; object storage (S3) for documents | Portable and reproducible |

Note the one trade-off versus an all-TypeScript stack: types aren't literally shared with the frontend. The fix is generating a typed TS client from FastAPI's OpenAPI schema, which keeps the contract in sync automatically.

### Architecture overview

A single **FastAPI service** owns all business logic and talks to PostgreSQL (with pgvector for embeddings). Both clients — Next.js web and React Native mobile — are thin consumers of that API via a generated typed client. Slow AI work (parsing, embedding, agent runs) is offloaded to **background workers** (Celery/RQ + Redis) so HTTP requests stay fast; results and streaming tokens flow back over Server-Sent Events or WebSockets. Cross-cutting concerns (auth, RBAC, rate limiting, logging) live as FastAPI dependencies/middleware so every route inherits them by default.

```
repo/
├─ apps/
│  ├─ api/            (FastAPI)
│  │  ├─ core/        auth, rbac, security, settings, theming
│  │  ├─ ai/          llm gateway, embeddings, rag, agents
│  │  └─ products/    document_reader (and future products)
│  ├─ worker/         (Celery/RQ tasks: ingest, embed, agent runs)
│  ├─ web/            (Next.js)
│  └─ mobile/         (React Native / Expo)
├─ packages/
│  └─ api-client/     (TS client generated from OpenAPI)
└─ infra/             (docker-compose, migrations, config)
```

The `core` and `ai` modules are the reusable base; `products/` is where a specific app (the document reader) lives. Cloning the base for a new product means keeping `core` + `ai` and swapping `products/`.

### Multi-tenancy decision

Decide early — it shapes the schema. Two options: **(A) One deployment per client** — simplest, strongest isolation, branding via a build-time/env config; best if clients are few and want data separation. **(B) Single deployment, many tenants** — a `tenant_id` on every row (or schema-per-tenant), theme/flags resolved per tenant at runtime; best for SaaS-style scale. Recommendation for a *base that you clone per product*: start with **(A)** and design the theme/config layer so it can graduate to **(B)** later (i.e., keep tenant-awareness in the config layer even if you don't enforce row-level tenancy yet).

### Data model (core entities)

The heart of the system is a users/roles/permissions triad plus supporting tables.

- **User** — id, email (unique), password_hash (nullable if using social/SSO), email_verified_at, status (active/suspended/pending), created/updated timestamps.
- **Profile** — user_id (1:1), display_name, avatar_url, bio, locale, timezone, and a JSONB `preferences` blob for extensibility.
- **Role** — id, key (e.g. `admin`, `user`), name, description, is_system (protect built-ins from deletion).
- **Permission** — id, key (e.g. `user:read`, `user:delete`, `role:assign`), description. Permissions are the atomic unit; roles bundle them.
- **RolePermission** — join table (role_id, permission_id).
- **UserRole** — join table (user_id, role_id); supports multiple roles per user.
- **RefreshToken / Session** — id, user_id, token_hash, device/user-agent, expires_at, revoked_at — for secure session management and remote logout.
- **AuditLog** — id, actor_user_id, action, target_type, target_id, metadata (JSONB), ip, created_at — records security-relevant events.
- **Setting** — key, value (JSONB), scope (global/tenant) — powers feature flags and configuration.
- **ThemeConfig** — logo_url, color tokens, font tokens, product name/copy (JSONB) — powers white-label branding.

AI-specific entities (used by the document reader and future AI products):

- **Document** — id, owner_user_id, title, source_type (pdf/docx/txt/url), storage_key (S3), status (uploaded/processing/ready/failed), page_count, created_at.
- **DocumentChunk** — id, document_id, chunk_index, text, token_count, metadata (JSONB: page, section). The unit that gets embedded.
- **Embedding** — chunk_id, vector (pgvector column), model, dim. Stored alongside the chunk (or in a dedicated vector table).
- **Conversation** — id, user_id, product, title, created_at — a chat session over one or more documents.
- **Message** — id, conversation_id, role (user/assistant/system/tool), content, citations (JSONB → chunk ids), token_usage, created_at.
- **AgentRun** — id, conversation_id, agent, status, steps (JSONB trace), cost, latency — observability + audit for agent executions.

Permissions-based RBAC (roles are collections of permissions) is preferred over role-only checks because it lets new products define custom roles without code changes — you check `can('user:delete')`, not `if role === admin`.

### API design

A versioned REST API (`/api/v1`) with consistent envelopes and predictable resource routes. Representative endpoints:

Authentication: `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `POST /auth/refresh`, `POST /auth/verify-email`, `POST /auth/forgot-password`, `POST /auth/reset-password`. Profile: `GET /me`, `PATCH /me`, `POST /me/avatar`, `PATCH /me/password`. Admin users: `GET /admin/users`, `POST /admin/users/invite`, `PATCH /admin/users/:id`, `POST /admin/users/:id/suspend`, `DELETE /admin/users/:id`. Roles & permissions: `GET /admin/roles`, `POST /admin/roles`, `PATCH /admin/roles/:id`, `POST /admin/users/:id/roles`. Config: `GET /config/theme`, `GET /config/flags`, `PATCH /admin/settings`.

Document reader / AI: `POST /documents` (upload → returns processing status), `GET /documents`, `GET /documents/:id`, `DELETE /documents/:id`, `POST /conversations`, `POST /conversations/:id/messages` (streams the answer via SSE), `GET /conversations/:id`, `POST /documents/:id/summarize`. AI endpoints that stream use Server-Sent Events (or WebSockets) so tokens and intermediate agent steps arrive incrementally.

Every non-public route passes through an auth dependency (valid access token) and an RBAC dependency (required permission). Validation uses Pydantic models, and FastAPI emits an OpenAPI schema that generates the typed TypeScript client for web and mobile — so client and server share one contract.

### Authentication & session flow

Use short-lived **access tokens** (JWT, ~15 min) plus long-lived **refresh tokens** (stored hashed in the DB, rotated on each use, revocable). Web stores tokens in httpOnly secure cookies to mitigate XSS token theft; mobile uses secure device storage (Keychain/Keystore). Passwords hashed with **Argon2id** (or bcrypt). Email verification and password reset use single-use, time-limited signed tokens. Build auth as a **pluggable module** so a product can swap the in-house implementation for a managed provider (Clerk/Auth0/Cognito) without rewriting consumers — the rest of the app only depends on "who is the current user and what can they do."

### RBAC enforcement

Define permissions as an enum/constants in the Python backend (single source of truth), surfaced to the clients through the generated API client. On the backend, a FastAPI dependency reads the route's required permission (e.g. `Depends(require_permission("user:delete"))`), loads the user's aggregated permissions, and allows/denies. On the frontend, the same permission set drives conditional UI (`<Can permission="user:delete">`) — but UI checks are convenience only; the server is the authority. Seed two system roles (`admin`, `user`) at bootstrap and protect them from deletion.

### Security layer

Security is treated as a first-class module, not an afterthought: Argon2id password hashing with a configurable strength policy; rate limiting on auth endpoints (and globally) to blunt brute-force and abuse; account lockout/backoff on repeated failures; refresh-token rotation with reuse detection; strict input validation and output encoding; security headers (CSP, HSTS, etc. via Helmet); CORS locked to known origins; secrets kept out of code (env/secret manager); audit logging of every security-relevant action; and dependency scanning in CI. Plan a security review at the end of Phase 5 (the `engineering:code-review` and `engineering:testing-strategy` skills can help structure it).

### White-label / theming layer

Branding is data, not code. A `ThemeConfig` record (colors, logo, fonts, product name, copy) is fetched at app startup and applied via design tokens — Tailwind CSS variables on web, a theme provider on mobile. Feature flags live in `Setting` records and gate modules/routes at runtime. The result: launching a new branded product is editing config + assets, not forking the repo. Keep all tenant-specific values behind this layer so the multi-tenant upgrade path stays open.

### AI layer — LLM gateway, embeddings & RAG

The AI layer is a reusable base module with three parts. The **LLM gateway** is a thin provider-agnostic interface (`generate`, `stream`, `embed`) wrapping whichever model provider you use, so products and agents never hardcode a vendor — swapping or A/B-testing models is a config change, and you get one place for retries, rate limits, cost tracking, and prompt logging. The **embeddings + vector-search service** turns text into vectors and runs similarity search over the vector DB (pgvector to start), exposing a simple `search(query, filters, k)` API. The **RAG pipeline** ties them together: a document is uploaded → parsed (PDF/DOCX/OCR as needed) → split into overlapping chunks → embedded → indexed with metadata; at query time the question is embedded, the top-k relevant chunks are retrieved (with metadata filtering and optional re-ranking), and those chunks are fed to the LLM as grounded context with citations back to the source.

Design choices that matter: chunking strategy (size + overlap, respecting document structure) drives answer quality; store rich chunk metadata (document, page, section) so answers can cite sources; keep ingestion asynchronous in the worker since embedding large documents is slow; and cache embeddings so re-processing is cheap.

### Multi-agent orchestration

Multi-agent work sits on top of the AI layer as its own framework so any product can define agents. The recommendation is **LangGraph** (or LlamaIndex's agent workflows) because it models agent execution as an explicit, inspectable graph rather than an opaque loop — which matters enormously for debugging and for the audit/observability you'll want in a production base. Core concepts: **agents** (a role + system prompt + a set of tools + a model), **tools** (typed functions agents can call — vector search, document fetch, calculators, external APIs), an **orchestrator/planner** that routes a request to the right agent(s) and can run them sequentially or in parallel, and **shared memory/state** passed between agents plus conversation history. Every run is recorded as an `AgentRun` with a step trace, token cost, and latency for observability and cost control. Intermediate steps stream to the client so users see progress rather than a spinner. Guardrails — max steps, token/cost budgets, tool allow-lists per agent, and timeouts — are enforced centrally so a runaway agent can't rack up cost or loop forever.

For the document reader, a minimal agent set is: a **retriever agent** (finds relevant chunks), an **answer agent** (composes a cited answer), and a **summarizer/extractor agent** (document summaries, key-fact or table extraction). Start simple — a single RAG-with-tools agent is often enough — and add specialized agents only when a task clearly needs them.

### First product — Document reader

The document reader is the first app on the base and the proof that the AI layer works. Users upload documents (PDF, DOCX, TXT, and later URLs), which are ingested asynchronously through the RAG pipeline and shown with a processing status. Once ready, the user opens a chat over one or more documents and asks questions; answers stream back with **inline citations** linking to the exact source chunk/page, so responses are verifiable rather than hallucinated. Additional actions include one-click document summaries and structured extraction (key facts, tables). All of this reuses the base's auth, RBAC (e.g. `document:read`, `document:delete`, per-user document ownership), profiles, theming, and observability — the product code is mostly the document UI, the chat experience, and the specific agents/prompts. Scope the MVP to: upload → ingest → chat with citations over a single document, then expand to multi-document collections, summaries, and extraction.

### Testing & quality

Layered strategy: unit tests for business logic (auth, RBAC resolution, validators); integration tests for API routes against a test database; end-to-end tests for the critical auth + admin flows; and contract tests ensuring the shared types/schemas match reality. Enforce lint, type-check, and test in CI on every PR. Because this is a base others will build on, prioritize test coverage of the security and RBAC core above feature breadth.

### Observability & ops

Structured logging with request correlation IDs, error tracking (e.g. Sentry) wired into all three apps, health/readiness endpoints, and database migration automation in the deploy pipeline. Containerize the API for reproducible deploys and keep environment parity between staging and production.

### Documentation (the part that makes it reusable)

A starter kit lives or dies on its docs. Ship: a "clone and rebrand in 30 minutes" quickstart, an architecture overview, the RBAC/permissions reference, the theming/feature-flag guide, and clearly marked extension points ("add a new module here," "add a new permission here"). The `engineering:documentation` skill can help produce these.

---

## Suggested next steps

1. Lock the open decisions: **multi-tenancy model** (per-deployment vs shared), **in-house auth vs managed provider**, and **LLM provider** (hosted API vs self-hosted/open-weight) — all cheaper to decide now than to migrate later.
2. Scaffold the repo and Phase 0 foundations (FastAPI + Postgres/pgvector + Redis + worker via docker-compose).
3. Build and secure the Phase 1 auth spine before anything else.
4. For the document reader specifically, prototype the RAG pipeline early (upload → chunk → embed → retrieve → cited answer) on a handful of real documents to validate chunking and answer quality before building the full UI.

I can turn any phase into a detailed task breakdown, scaffold the actual repo structure, produce a proof-of-concept RAG pipeline, or output this as a formatted Word/PDF — just say which.
