#!/usr/bin/env bash
# The verification gate. One definition, used by humans, the /verify skill, and
# CI — so "it passed locally" and "it passed in CI" mean the same thing.
#
#   ./scripts/verify.sh                 fast checks; needs nothing running
#   ./scripts/verify.sh --integration   adds tests that need the data services
#   ./scripts/verify.sh --e2e           adds Playwright; needs the full stack
#   ./scripts/verify.sh --all           everything
#
# Every step runs even if an earlier one fails, so you get the complete picture
# in one pass instead of fixing one thing at a time.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

RUN_INTEGRATION=0
RUN_E2E=0
for arg in "$@"; do
    case "$arg" in
        --integration) RUN_INTEGRATION=1 ;;
        --e2e) RUN_E2E=1 ;;
        --all) RUN_INTEGRATION=1; RUN_E2E=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# Prefer the project venv; fall back to whatever python is active so this works
# in CI, where dependencies are installed into the ambient environment.
if [[ -x "$BACKEND/.venv/bin/python" ]]; then
    PY="$BACKEND/.venv/bin"
else
    PY=""
fi
py() { if [[ -n "$PY" ]]; then "$PY/$1" "${@:2}"; else "$@"; fi; }

FAILED=()
PASSED=()

step() {
    local name="$1"; shift
    printf '\n\033[1m▶ %s\033[0m\n' "$name"
    if "$@"; then
        PASSED+=("$name")
    else
        FAILED+=("$name")
        printf '\033[31m✗ %s failed\033[0m\n' "$name"
    fi
}

# ── Backend ───────────────────────────────────────────────────────────────────
cd "$BACKEND" || exit 1
step "backend: ruff"   py ruff check .
step "backend: mypy"   py mypy app
step "backend: pytest" py pytest -q

if [[ $RUN_INTEGRATION -eq 1 ]]; then
    # These need Postgres, Redis, and Qdrant. Fail loudly rather than silently
    # skipping — a green run that quietly tested nothing is the worst outcome.
    step "backend: pytest -m integration" py pytest -q -m integration
fi

# ── Frontend ──────────────────────────────────────────────────────────────────
cd "$FRONTEND" || exit 1
if [[ ! -d node_modules ]]; then
    echo "node_modules missing — running npm install"
    npm install --silent
fi
step "frontend: typecheck" npm run --silent typecheck
step "frontend: build"     npm run --silent build
# Advisories are a supply-chain signal for a base others will build on, so a
# high-severity finding fails the gate rather than printing a warning.
step "frontend: audit"     npm audit --audit-level=high

if [[ $RUN_E2E -eq 1 ]]; then
    step "frontend: playwright" npx playwright test
fi

# ── Summary ───────────────────────────────────────────────────────────────────
printf '\n\033[1m── verify ──\033[0m\n'
for name in "${PASSED[@]}"; do printf '\033[32m  ✓ %s\033[0m\n' "$name"; done
for name in "${FAILED[@]:-}"; do
    [[ -n "$name" ]] && printf '\033[31m  ✗ %s\033[0m\n' "$name"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '\n\033[31m%d check(s) failed.\033[0m\n' "${#FAILED[@]}"
    exit 1
fi
printf '\n\033[32mAll checks passed.\033[0m\n'
