# White-Label App

A reusable platform base — authentication, profiles, RBAC, an admin module, a
theming/feature-flag layer, and an AI/agents layer — that gets cloned to start a
new product. The first product built on it is a document reader (chat and Q&A
over your documents).

Full roadmap and technical spec: [`docs/white-label-app-plan.md`](docs/white-label-app-plan.md).

**Current state: Phase 0 (foundations).** The stack boots, migrations run, the
API reports the health of every dependency, and the web client renders it.
There is no authentication yet — see [Phase status](#phase-status).

---

## Quickstart

Your code runs on your host; Docker only supplies the three servers the app
talks to. Requires Python 3.12+, Node 20+, and Docker with Compose v2.

```bash
# 1. Data services only — Postgres, Redis, Qdrant
cp infra/.env.example infra/.env          # defaults work as-is
docker compose -f infra/docker-compose.yml up -d

# 2. API, on the host
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload

# 3. Web client, in a third terminal
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

### Why Docker is only used for the data services

Because that is the only part it earns. Running the API in a container makes the
dev loop worse — the debugger doesn't attach cleanly through a volume mount and a
dependency change becomes an image rebuild. Running Postgres, Redis, and Qdrant
in containers makes it better: per-project versions instead of one shared
Homebrew install, and `docker compose down -v` wipes all state in seconds. That
last part matters a lot during the auth and RBAC phases, when you will re-run
migrations from an empty database over and over.

If you would rather not use Docker at all, the app doesn't care — point
`DATABASE_URL`, `REDIS_URL`, and `QDRANT_URL` at Homebrew installs or at managed
free tiers (Neon, Upstash, Qdrant Cloud) and everything works unchanged.

A containerised API is still defined for parity checks and as the deploy
artifact, behind a profile so it stays out of the default path:

```bash
docker compose -f infra/docker-compose.yml --profile full up --build
```

Then open:

| URL | What it is |
|---|---|
| http://localhost:3000 | Web client — status page showing every dependency |
| http://localhost:8000/docs | Interactive API docs (local environment only) |
| http://localhost:8000/health | Liveness — touches nothing |
| http://localhost:8000/health/ready | Readiness — per-dependency status and latency |

The page at :3000 showing three green dots means the foundation is correct end
to end: browser → CORS → FastAPI → Postgres/Redis/Qdrant.

### Backend checks

```bash
cd backend
pytest                    # unit tests — need no running services at all
pytest -m integration     # readiness test — needs the data services up
ruff check . && mypy app
```

---

## Layout

```
backend/
  app/
    main.py            App factory: middleware, CORS, error handlers, routers
    api.py             Versioned router assembly — the one list of what's exposed
    core/              Cross-cutting: config, db, clients, errors, logging
    modules/           The reusable base, one vertical slice per folder
      health/          Liveness + readiness probes
      meta/            Deployment identity (first white-label seam)
    products/          Product-specific code (document reader) — not yet created
  alembic/             Migrations
  tests/
frontend/
  src/app/             Next.js App Router pages
  src/lib/api.ts       API client — replaced by an OpenAPI-generated one later
  src/theme/tokens.ts  Design tokens: plain data, shared with mobile eventually
  e2e/                 Playwright specs
infra/
  docker-compose.yml   Local stack
scripts/verify.sh      The verification gate — used by humans, /verify, and CI
.claude/
  agents/              test-author, tenancy-reviewer, e2e-explorer
  skills/verify/       The /verify command
.github/workflows/     CI
CLAUDE.md              Architecture contract every agent reads before writing code
docs/
  white-label-app-plan.md
```

### Conventions worth knowing before you add code

**Modules are vertical slices.** A module owns its `router.py`, `schemas.py`,
`models.py`, and `service.py`. Nothing is organized by layer (no top-level
`models/` holding every model), because a feature flag has to be able to switch
off a whole module — and that is impossible when a feature is smeared across six
directories.

**Register routers in one place.** `app/api.py` is the only file that mounts
module routers. Read it to know what the API exposes.

**Register models in one place.** `app/core/models.py` imports every model so
Alembic's autogenerate can see them. A model missing from that file produces
migrations that silently omit its table.

**Branding is data.** No product name, colour, or copy is hardcoded. The
backend serves identity from config via `/api/v1/meta`; the frontend styles
itself from CSS custom properties that `applyTheme()` can rewrite at runtime.
Adding a hardcoded brand string anywhere is the one thing that breaks the point
of this repo.

**Errors have one shape.** Service code raises `AppError` subclasses
(`NotFoundError`, `ConflictError`, `UnauthorizedError`, `ForbiddenError`); the
handlers in `core/errors.py` render them as
`{"error": {code, message, details, request_id}}`. Never return a bare dict for
a failure.

**Every response carries `X-Request-ID`.** It appears in every log line for that
request. When something breaks, that ID is how you find it.

---

## Testing

One command runs everything:

```bash
./scripts/verify.sh                 # lint, types, unit tests, build, audit
./scripts/verify.sh --integration   # adds tests needing the data services
./scripts/verify.sh --e2e           # adds Playwright; needs the full stack
./scripts/verify.sh --all
```

Same script runs in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)),
so "passes locally" and "passes in CI" mean the same thing.

### Layers

| Layer | Location | Needs | Covers |
|---|---|---|---|
| Unit | `backend/tests/` | nothing | Service logic, validators, RBAC resolution |
| Integration | `backend/tests/`, `-m integration` | Postgres, Redis, Qdrant | Routes against a real database, migrations |
| E2E | `frontend/e2e/` | full stack | Flows a user actually performs |

Every route gets four integration tests, because these are the ones that get
skipped and matter most in a multi-customer product: happy path; unauthenticated
→ 401; authenticated without the permission → 403; **authenticated as a different
organization → 404** — not 403, since 403 confirms the resource exists.

### The agents

Defined in [`.claude/agents/`](.claude/agents/), driven by the conventions in
[`CLAUDE.md`](CLAUDE.md):

| Agent | Use it when |
|---|---|
| `test-author` | You added or changed a module and it needs coverage |
| `tenancy-reviewer` | Before merging anything touching models, queries, routes, or auth |
| `e2e-explorer` | A user-facing flow landed, or before a release |

The division of labour that makes this work: **agents author tests, CI runs
them.** Agent-written tests are committed and then execute deterministically. An
agent that re-derives assertions on every CI run is slow, costs money per run,
and can disagree with itself between runs — that is a worse test suite, not a
better one. Agents are for the expensive judgment (what should this module's
tests cover, given how the rest of the repo is tested); `pytest` and `playwright`
are for the gate.

## Phase status

A phase ships only after it passes the gate in [`CLAUDE.md`](CLAUDE.md#the-phase-gate):
`./scripts/verify.sh --all` green, permission and cross-tenant tests written,
flows driven in a real browser, tenancy review done, this table updated. No phase
starts before the previous one clears it.


| Phase | Scope | State |
|---|---|---|
| 0 | Foundations: compose stack, config, DB session, migrations, health, error contract, logging, web shell | **Done** |
| 1 | Auth spine: register, verify email, login/logout, refresh rotation, password reset | Next |
| 2 | Profiles & RBAC: profile CRUD, avatar, roles, permissions, `require_permission` | Not started |
| 3 | Admin module: user management, role assignment, audit log viewer | Not started |
| 4 | White-label layer: `ThemeConfig`, feature flags, settings framework | Not started |
| 5 | AI foundation: LLM gateway, embeddings + vector search, RAG ingestion pipeline | Not started |
| 6 | Multi-agent orchestration: agents, tools, planner, step streaming, guardrails | Not started |
| 7 | First product: document reader (upload, chat with citations, summaries) | Not started |
| 8 | Hardening: security review, rate limiting, audit completeness, a11y, docs | Not started |

## Decisions

**Made**

- **Python/FastAPI backend, Next.js web client.** The product is AI-heavy and
  that ecosystem lives in Python. The cost is that types aren't literally shared
  with the frontend; the fix is generating a TS client from the OpenAPI schema.
- **Web first; mobile later.** React Native screens have to be written
  separately regardless of when they start, so deferring mobile costs no UI
  work. Design tokens are kept as framework-free data from day one so the
  eventual mobile client shares them.
- **Qdrant for vectors, Postgres for everything relational.** The plan's
  pgvector-first suggestion is reasonable, but Qdrant is where this is heading
  and running both from the start avoids a migration.
- **Async Alembic.** Migrations use the same `+asyncpg` driver as the app, so
  there is one Postgres driver rather than two.
- **`postcss` and `sharp` are pinned via npm `overrides`.** Next 16.2.12 still
  resolves versions with three high-severity advisories, and `npm audit fix`
  "solves" it by proposing a downgrade to Next 9. The overrides pull in the
  patched releases instead; `npm audit` is clean. Remove them once a Next
  release ships the fixed ranges itself.
- **Docker runs the data services, not the app.** Containers give per-project
  database versions and instant state resets; containerising the API would only
  cost debugger ergonomics and rebuild time. The API image still exists for
  parity and deployment, behind the `full` profile.
- **Migrations are a separate step.** `alembic upgrade head`, run by hand locally
  and as a pipeline step in deploys — never from the app's start command, where
  concurrent replicas would race each other.

- **Multi-tenant schema, single-tenant deployment.** Every tenant-owned table
  carries `organization_id` from the first domain migration, and Postgres
  Row-Level Security enforces it at the database. But the first customer gets
  their own deployment — strongest isolation, simplest ops, and the easiest
  answer to a security review.

  The two halves are deliberately decoupled. Adding `organization_id` now costs a
  column, an index, and a filter in one shared helper. Adding it later costs a
  data migration plus an audit of every query ever written, where a single missed
  filter leaks one customer's data to another. RLS is the safety net: a query
  that forgets its tenant context returns nothing rather than everything.
  Consolidating into shared multi-tenant SaaS later is then a routing and config
  change, not a schema rewrite.

- **In-house authentication behind a pluggable seam.** Argon2id, short-lived
  access tokens, rotating hashed refresh tokens with reuse detection. Free, and
  no per-customer vendor account to configure each time the base is cloned.
  Consumers depend only on `get_current_user` and `require_permission`, so if
  enterprise SSO/SAML forces a managed provider later, it swaps underneath
  without touching call sites.

- **Agents author tests; CI runs them.** Test code is generated with agent help,
  then committed and executed deterministically. Agents are not the test suite —
  an agent that re-derives assertions on every run is slow, costs money per run,
  and can disagree with itself between runs. See [Testing](#testing).

**Still open — decide before the phase that depends on it**

- **LLM provider** — hosted API vs. self-hosted open-weight. *Blocks Phase 5.*

## Known gaps in Phase 0

Called out so they aren't mistaken for finished work:

- **No authentication.** Every endpoint is public. Nothing here is safe to
  expose beyond localhost.
- **No rate limiting**, no security headers (CSP/HSTS), no audit logging.
- **Postgres and Redis are pinned by major version only.** Pin exact patch
  versions before this touches staging. Qdrant is already pinned exactly —
  `qdrant/qdrant:latest` on Docker Hub lags real releases, and the client rejects
  more than a minor version of drift, so its tag must track
  `qdrant-client` in `backend/pyproject.toml`.
- **CI is written but dormant.** GitHub only reads workflows from a repository
  root, and this currently lives inside a parent repo. It activates when
  `White-Label-App/` becomes its own repository.
- **The E2E suite is thin by design.** All four specs pass against the running
  stack, but there is only one page to test until Phase 1 lands real flows.
- **`src/lib/api.ts` is hand-written.** It gets replaced by an OpenAPI-generated
  client, at which point the response types stop being able to drift from the
  Python models.
