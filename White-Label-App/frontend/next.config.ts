import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Fail the build on type errors rather than shipping them — the default
  // already does this, but stating it prevents a future "just ignore it" edit.
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
