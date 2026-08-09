---
name: verify
description: Run the full verification gate for the White-Label App — backend lint, type-check and tests, frontend type-check, build and dependency audit, optionally integration and Playwright E2E. Use before claiming any work is complete, and after any change to backend/ or frontend/.
---

# Verify

Runs every check that guards this repo. Use it instead of running the individual
commands, so nothing gets skipped.

## Run it

```bash
./scripts/verify.sh                 # fast — needs nothing running
./scripts/verify.sh --integration   # adds tests needing Postgres/Redis/Qdrant
./scripts/verify.sh --e2e           # adds Playwright; needs the full stack
./scripts/verify.sh --all           # everything
```

Pick the level that matches what changed:

- Touched backend logic or frontend code → default run.
- Touched models, migrations, queries, or routes → `--integration`. Unit tests do
  not talk to a database, so they cannot catch a broken migration or a missing
  tenant filter.
- Touched a user-facing flow → `--e2e`.

For `--integration` and `--e2e` the data services must be up:

```bash
docker compose -f infra/docker-compose.yml up -d
```

## Interpreting the result

Every step runs even after an earlier failure, so the summary is the full
picture. Work the failures in this order — earlier ones often cause later ones:

1. **ruff** — formatting and lint. Usually `ruff check --fix .`.
2. **mypy** — runs strict. A new `Any` or a missing return type will fail it.
   Do not silence it with `# type: ignore` without a comment saying why.
3. **pytest** — a real failure. Read the assertion before touching the test:
   the default assumption is that the code is wrong, not the test.
4. **typecheck / build** — the Next build also type-checks, so a build failure
   after a clean typecheck usually means a server/client boundary mistake.
5. **audit** — a high-severity advisory. Pin the patched transitive version in
   `package.json` `overrides`. Never `npm audit fix --force`; it downgrades
   majors to "resolve" advisories.

## Rules

- Report the actual result. If a step fails, say so and show the output — never
  describe the gate as passing when it did not.
- Never weaken a check to make it pass: do not delete an assertion, add a blanket
  `# type: ignore`, lower `--audit-level`, or mark a failing test `skip` to get
  green. If a check is genuinely wrong, say why and propose the fix.
- A skipped step is not a passed step. If the data services were down and
  integration tests did not run, state that explicitly.
