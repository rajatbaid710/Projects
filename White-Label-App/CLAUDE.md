# Working in this repo

A reusable platform base (auth, profiles, RBAC, admin, theming, AI layer) that
gets cloned to start new products. Python/FastAPI backend, Next.js web client.
Roadmap: `docs/white-label-app-plan.md`. Current phase and open decisions:
`README.md`.

This file is the architecture contract. Code that violates it is wrong even if
it works, because the value of this repo is that the next product can be built
from it without re-litigating decisions.

## Commands

```bash
# From repo root
docker compose -f infra/docker-compose.yml up -d    # Postgres, Redis, Qdrant

# Backend (backend/, with .venv activated)
uvicorn app.main:app --reload
alembic upgrade head
alembic revision --autogenerate -m "description"
pytest                    # unit — needs no services
pytest -m integration     # needs the data services running
ruff check . && mypy app

# Frontend (frontend/)
npm run dev
npm run typecheck && npm run build
npx playwright test       # E2E — needs the full stack running
```

Run `/verify` to execute the whole gate at once. Do that before claiming work is
complete.

## Non-negotiables

**1. Every tenant-owned row carries `organization_id`, and every query filters on
it.** This product is sold to multiple companies. A query missing its tenant
filter is a data leak between customers, not a bug. Use the shared tenant-scoped
query helper rather than writing raw filters, and rely on Postgres RLS as the
backstop — never as the only defence.

**2. Every non-public route declares its permission.** Routes take
`Depends(require_permission("resource:action"))`. Never check `if user.role ==
"admin"` — roles are bundles of permissions, and products define their own roles.
Frontend permission checks are convenience only; the server is the authority.

**3. Branding is data, never code.** No product name, colour, logo, or customer
name is hardcoded anywhere. The backend serves identity from config; the frontend
styles itself from CSS custom properties that `applyTheme()` rewrites at runtime.
A hardcoded brand string defeats the purpose of the repo.

**4. Errors use the envelope.** Raise `AppError` subclasses (`NotFoundError`,
`ConflictError`, `UnauthorizedError`, `ForbiddenError`) from service code. The
handlers in `core/errors.py` render `{"error": {code, message, details,
request_id}}`. Never return a bare dict or raw string for a failure, and never
put internal detail (SQL, paths, stack frames) in a client-visible message.

**5. Secrets come from the environment.** No key, token, or password in code,
tests, fixtures, or committed config. Add new settings to `core/config.py` and
document them in `infra/.env.example`.

## Structure

Backend modules are **vertical slices**, not layers. A module owns its own
`router.py`, `schemas.py`, `models.py`, `service.py`. There is no top-level
`models/` holding every model. This exists so a feature flag can disable a whole
module — impossible when a feature is spread across six directories.

```
backend/app/
  main.py       App factory. Middleware, CORS, handlers, router mounting.
  api.py        THE list of what the API exposes. Every router registers here.
  core/         Cross-cutting only: config, db, clients, deps, errors, logging.
  modules/      The reusable base. One folder per vertical slice.
  products/     Product-specific code (document reader). Never imported by core.
```

Two registries that break silently when you forget them:

- **`app/api.py`** — the only place routers are mounted. A router not registered
  here does not exist.
- **`app/core/models.py`** — imports every model so Alembic can see it. A model
  missing here produces migrations that silently omit its table.

Dependency direction is one-way: `products/` may import from `core/` and
`modules/`; `core/` and `modules/` must never import from `products/`.

## Adding a module

1. Create `modules/<name>/` with `router.py`, `schemas.py`, `models.py`,
   `service.py`.
2. Models: inherit `Base`, include `organization_id` if tenant-owned.
3. Register the model in `core/models.py`, then
   `alembic revision --autogenerate -m "add <name>"`. **Read the generated
   migration** — autogenerate misses RLS policies, index intent, and data
   backfills.
4. Routes: Pydantic schemas in and out, `require_permission` on every non-public
   route, service layer does the work (routers stay thin).
5. Register the router in `api.py`.
6. Tests — see below.
7. Run `/verify`.

## Testing

Agents write tests; the tests are committed and CI runs them deterministically.
Do not build anything that calls a model at assertion time.

Every module needs:

- **Unit tests** for service logic, with no database.
- **Integration tests** per route, marked `@pytest.mark.integration`, covering
  the four cases that get skipped and matter most:
  1. the happy path,
  2. **unauthenticated** → 401,
  3. **authenticated but lacking the permission** → 403,
  4. **authenticated as a different organization** → 404 (not 403 — a wrong-tenant
     request must not confirm the resource exists).
- **E2E** (`frontend/e2e/`) only for flows a user actually performs end to end.

Tests assert on the error envelope's `code`, not on message text — messages are
copy and will change.

## The phase gate

**A phase is not finished until it has been tested. Do not start the next phase
before the current one passes this gate.**

1. `./scripts/verify.sh --all` is green — lint, types, unit, integration, build,
   audit, E2E. Not a subset.
2. Every route added in the phase has its four integration tests (happy path,
   401, 403, wrong-organization → 404).
3. The phase's user-facing flows have been **driven in a real browser**, not just
   asserted in a test — and the result looked correct.
4. `tenancy-reviewer` has reviewed the phase diff if it touched models, queries,
   routes, or auth.
5. `README.md` phase status is updated, and anything knowingly incomplete is
   listed under Known gaps.

Report the gate honestly. A phase where step 1 passed but step 3 was skipped is
not done — say which steps ran and which did not. Never describe a phase as
complete on the strength of a green unit suite alone: unit tests do not talk to a
database, so they cannot catch a broken migration, a missing tenant filter, or a
route that was never registered.

## Style

Python: 3.12+, full type hints (`mypy` runs strict), `async def` for anything
touching I/O, Google-style docstrings on non-obvious functions. Ruff is the
formatter and linter.

TypeScript: strict mode, no `any`, `type` over `interface` for object shapes.

Comments explain *why*, not *what*. A comment restating the code is noise; a
comment recording why an approach was rejected is worth keeping.

## Don't

- Don't hardcode a customer or product name anywhere.
- Don't add a route without a permission dependency unless it is genuinely public
  (auth entry points, health, meta) — and say so in a comment when it is.
- Don't write a query against a tenant-owned table without a tenant filter.
- Don't commit a migration you have not read line by line.
- Don't use `npm audit fix --force` — it "fixes" by downgrading majors. Pin the
  patched transitive version in `overrides` instead.
- Don't run migrations from an application start command; replicas race.
- Don't put anything secret behind a `NEXT_PUBLIC_` variable — it ships to the
  browser.
