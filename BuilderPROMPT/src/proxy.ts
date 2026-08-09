import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { getSessionFromRequest } from "@/lib/auth/session";

const PUBLIC_API_PREFIXES = ["/api/auth/"];
const PUBLIC_PAGES = ["/login"];

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  const isPublicApi = PUBLIC_API_PREFIXES.some((p) => pathname.startsWith(p));
  const isApi = pathname.startsWith("/api/");
  const isPublicPage = PUBLIC_PAGES.includes(pathname);

  if (isPublicApi) return NextResponse.next();

  const session = await getSessionFromRequest(request);

  if (isApi) {
    if (!session) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }
    return NextResponse.next();
  }

  // Page routes
  if (!session && !isPublicPage) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (session && isPublicPage) {
    return NextResponse.redirect(new URL("/scan", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|manifest.webmanifest|icons/|sw.js).*)",
  ],
};
