#!/usr/bin/env bash
# Starts the Next.js dev server and a Cloudflare quick tunnel together, then
# prints the public https://*.trycloudflare.com URL you can open on your
# phone. Ctrl+C stops both. See docs/HOSTING.md for a permanent tunnel setup.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-3000}"
DEV_LOG="$(mktemp -t billbox-dev)"
TUNNEL_LOG="$(mktemp -t billbox-tunnel)"

TUNNEL_PID=""

cleanup() {
  echo ""
  echo "Stopping BillBox and the tunnel…"
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  pkill -f "cloudflared tunnel --url http://localhost:$PORT" 2>/dev/null || true
  lsof -ti:"$PORT" | xargs kill -9 2>/dev/null || true
  rm -f "$DEV_LOG" "$TUNNEL_LOG"
}
trap cleanup EXIT INT TERM

# Free the port in case a previous run was left dangling.
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "Port $PORT is already in use — freeing it…"
  lsof -ti:"$PORT" | xargs kill -9 2>/dev/null || true
  sleep 1
fi

echo "Starting the app on http://localhost:$PORT …"
npm run dev -- --port "$PORT" >"$DEV_LOG" 2>&1 &

echo -n "Waiting for it to come up"
for _ in $(seq 1 30); do
  if curl -s -o /dev/null "http://localhost:$PORT"; then
    echo " done."
    break
  fi
  echo -n "."
  sleep 1
done
if ! curl -s -o /dev/null "http://localhost:$PORT"; then
  echo ""
  echo "The app didn't come up in time. Last log lines:"
  tail -n 20 "$DEV_LOG"
  exit 1
fi

echo "Opening a Cloudflare tunnel…"
npx --yes cloudflared tunnel --url "http://localhost:$PORT" >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

TUNNEL_URL=""
echo -n "Waiting for the public URL"
for _ in $(seq 1 60); do
  TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -n 1 || true)
  [ -n "$TUNNEL_URL" ] && { echo " done."; break; }
  echo -n "."
  sleep 1
done

echo ""
echo "======================================================"
if [ -n "$TUNNEL_URL" ]; then
  echo "  BillBox is live!"
  echo ""
  echo "  On this computer:  http://localhost:$PORT"
  echo "  On your phone:     $TUNNEL_URL"
  echo ""
  echo "  This URL changes every time you run this script."
  echo "  For a permanent URL, see docs/HOSTING.md."
else
  echo "  BillBox is running at http://localhost:$PORT"
  echo "  The tunnel URL didn't appear yet — check $TUNNEL_LOG"
fi
echo "======================================================"
echo ""
echo "Press Ctrl+C to stop."

wait
