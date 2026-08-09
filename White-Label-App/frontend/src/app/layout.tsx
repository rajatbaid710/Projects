import type { Metadata } from "next";

import "./globals.css";

/**
 * The page title comes from the API, not from a constant in this repo — the
 * same white-label rule the backend follows. If the API is unreachable (a very
 * common state during local development) we fall back rather than failing the
 * render, because a broken backend should show the status page, not a crash.
 */
export async function generateMetadata(): Promise<Metadata> {
  const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  try {
    const response = await fetch(`${base}/api/v1/meta`, { cache: "no-store" });
    if (response.ok) {
      const meta = (await response.json()) as { product_name: string };
      return { title: meta.product_name };
    }
  } catch {
    // fall through to the default below
  }
  return { title: "White Label App" };
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
