# BillBox

A mobile-first PWA for capturing invoices and billing documents — printed or
handwritten — and automatically extracting the key GST-relevant fields with
Claude, ready to review on your phone. Built to be self-hosted on one
machine at essentially zero fixed cost.

## Stack

- **Next.js 16** (App Router, TypeScript) — single full-stack app
- **Tailwind CSS v4 + shadcn/ui** — mobile-first components
- **SQLite via Drizzle ORM** (`better-sqlite3`) — zero-ops database, one file
- **Local filesystem** — uploaded documents stored under `data/uploads/`
- **Claude API** (`@anthropic-ai/sdk`) — server-side extraction agent, structured output
- **`jose`** — signed session cookies (email-only sign-in, no passwords)
- **`sharp` / `heic-convert`** — image downscaling and iPhone HEIC support

See [`INVOICE_APP_BUILD_PROMPT.md`](./INVOICE_APP_BUILD_PROMPT.md) for the full original spec.

## Setup

```bash
npm install
```

Copy `.env.example` values into `.env.local` (already created for local dev
with a generated `SESSION_SECRET`) and set:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
Without it, uploads still work but every extraction fails with a clear
"not configured yet" error — everything else in the app (auth, history, UI)
works fine so you can explore it before adding a key.

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The database
(`data/app.db`) and uploaded files (`data/uploads/`) are created
automatically on first run — no separate migration step.

**To use the app from your phone**, run this instead — it starts the server
and prints a public URL:

```bash
npm run tunnel
```

See [`docs/HOSTING.md`](./docs/HOSTING.md) for a permanent URL and other details.

## How it works

1. **Sign in** with just your email — no password, no verification code.
   The email is only used to keep every user's uploads and history separate.
2. **Scan** — photograph or upload a bill. It's downscaled, sent to Claude
   with a GST-aware structured-output schema, and the result is saved.
3. **Review** — every extracted field is editable; anything Claude wasn't
   confident about is highlighted; a totals check warns if the numbers
   don't add up. Confirming saves it to your history.
4. **History** — search, filter by status or date, see a running monthly
   total, and revisit or delete any document.

Extraction runs as a background task on the Node process serving the app
(not a queue) — the upload request returns immediately and the client polls
for status, so multiple uploads don't block each other or the UI.

## Model choice & cost

`EXTRACTION_MODEL` (default `claude-sonnet-5`) controls which model reads
your documents. Approximate cost per document at default settings is a few
cents; the exact figure is tracked per-extraction and totaled on the
**Settings** page. Swap to `claude-haiku-4-5` for a cheaper/faster model on
mostly-printed invoices, or `claude-opus-5` for the hardest handwriting.

## Project structure

```
src/
  app/                    routes (App Router)
    (app)/                 authenticated pages: scan, history, documents/[id], settings
    api/                    route handlers: auth, documents, usage
    login/                  email-only sign-in
  components/              UI (shadcn primitives + feature components)
  lib/
    auth/                   session cookies (jose) + email-based user lookup
    db/                     Drizzle schema + SQLite client (self-bootstrapping)
    documents/               shared serialization + the extraction pipeline
    extraction/              Zod schema + the Claude API call
    upload/                  file validation, HEIC conversion, downscaling
  proxy.ts                 route protection (Next.js 16's middleware replacement)
scripts/generate-icons.mjs      regenerates the PWA/app icons from one SVG
scripts/start-with-tunnel.sh    `npm run tunnel` — dev server + public URL
```

## Known simplifications

- No offline mode or service worker — deliberately out of scope per the
  original spec (a flaky connection just means a failed upload you can retry).
- Tally integration is a **future phase**, not built here. The extraction
  schema already carries every field a Tally purchase voucher needs
  (GSTIN, HSN/SAC, CGST/SGST/IGST split, place of supply) so nothing needs
  to change in the data model when that phase starts.
- Single-process, single-machine hosting only (see `docs/HOSTING.md`) — this
  is a personal/small-business tool, not built for concurrent heavy load.
