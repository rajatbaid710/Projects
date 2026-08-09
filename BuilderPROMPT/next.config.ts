import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next.js dev mode blocks cross-origin requests to dev-only resources
  // (e.g. the HMR websocket) by default. This app is designed to be opened
  // from a Cloudflare Tunnel origin (see docs/HOSTING.md), which is a
  // different origin than localhost, so it needs to be allow-listed.
  // Quick tunnels get a random *.trycloudflare.com subdomain on every run,
  // hence the wildcard rather than a fixed hostname.
  allowedDevOrigins: ["*.trycloudflare.com"],
};

export default nextConfig;
