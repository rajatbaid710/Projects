# Hosting BillBox and reaching it from your phone

BillBox runs as a single Node.js process on your machine (a laptop, mini PC,
or home server). Cloudflare Tunnel gives that local server a public HTTPS
URL that works from any phone, on any network — no router configuration, no
port forwarding.

## 1. Fastest path: one command

```bash
npm run tunnel
```

This starts the dev server, opens a Cloudflare quick tunnel, waits for both
to come up, and prints the public URL:

```
======================================================
  BillBox is live!

  On this computer:  http://localhost:3000
  On your phone:     https://random-words-here.trycloudflare.com

  This URL changes every time you run this script.
  For a permanent URL, see docs/HOSTING.md.
======================================================
```

Open that URL on your phone — that's the whole setup. Press `Ctrl+C` to stop
both the server and the tunnel together. The URL changes every time you run
the script, so this is for daily/casual use, not something you'd bookmark
long-term (see the permanent option below for that). The script lives at
`scripts/start-with-tunnel.sh` if you want to see or tweak how it works.

### Manual equivalent

If you'd rather run the two pieces yourself (e.g. to use a production build
instead of dev mode):

```bash
npm run build && npm run start   # in one terminal
npx cloudflared tunnel --url http://localhost:3000   # in another
```

The second command prints the same kind of random `trycloudflare.com` URL.

## 2. Permanent tunnel — recommended for daily use

Requires a free Cloudflare account and a domain added to it (a cheap or free
subdomain works fine — you don't need to route real traffic through
Cloudflare's CDN for this, just use it as the tunnel's DNS).

```bash
# One-time setup
cloudflared tunnel login
cloudflared tunnel create billbox
cloudflared tunnel route dns billbox app.yourdomain.com
```

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: billbox
credentials-file: /Users/you/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: app.yourdomain.com
    service: http://localhost:3000
  - service: http_status:404
```

Then run it:

```bash
cloudflared tunnel run billbox
```

Your phone (or anyone's) can now reach `https://app.yourdomain.com` — signed
in with the same email flow as local `localhost:3000`, from anywhere.

### Keep it running across reboots (macOS)

```bash
sudo cloudflared service install
```

This installs `cloudflared` as a launchd service that starts automatically.
Do the same for the app itself with your process manager of choice (`pm2`,
a `launchd` plist, or just a `screen`/`tmux` session — anything that keeps
`npm run start` alive).

## Why cookies still work behind the tunnel

Cloudflare Tunnel terminates TLS at Cloudflare's edge and forwards plain
HTTP to your local server, but the browser only ever sees the `https://`
origin — the session cookie is set with `Secure`, which the browser honors
correctly here since the page itself loaded over HTTPS. No extra
configuration is needed on the app side.

## Security notes

- **Sign-in is email-only, with no verification code.** Typing any email
  address is enough to create an account and start uploading under that
  name — this app doesn't confirm the person typing it actually owns that
  inbox. That's a reasonable tradeoff for a small trusted team sharing one
  tunnel URL, but it means the tunnel URL itself is your real access
  boundary, not the login screen.
- Because of that, it's worth putting **Cloudflare Access** (free for small
  teams) in front of the tunnel hostname so only people you've approved can
  even reach the login screen — see Cloudflare's Zero Trust docs. Treat the
  tunnel URL itself as sensitive, the same way you'd treat a shared password.
- Don't commit `.env.local` or the `data/` directory (both are gitignored by
  default) — they contain your session secret and every uploaded document.
