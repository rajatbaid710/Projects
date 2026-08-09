import type { MetadataRoute } from "next";

// This app is tunneled to a public URL (see docs/HOSTING.md) but holds
// private financial documents — keep it out of search engines.
export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", disallow: "/" },
  };
}
