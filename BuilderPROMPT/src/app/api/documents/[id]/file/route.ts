import { NextResponse } from "next/server";
import fs from "fs/promises";
import { and, eq } from "drizzle-orm";
import { requireSession } from "@/lib/auth/session";
import { db } from "@/lib/db";
import { documents } from "@/lib/db/schema";
import { resolveUploadPath } from "@/lib/upload/process";

type RouteParams = { params: Promise<{ id: string }> };

export async function GET(request: Request, { params }: RouteParams) {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const { id } = await params;
  const [document] = await db
    .select()
    .from(documents)
    .where(and(eq(documents.id, id), eq(documents.userId, session.sub)))
    .limit(1);

  if (!document) return NextResponse.json({ error: "Document not found" }, { status: 404 });

  const wantsOriginal = new URL(request.url).searchParams.get("variant") === "original";
  const useOriginal = wantsOriginal || !document.previewPath;
  const relativePath = useOriginal ? document.storedPath : document.previewPath!;
  const contentType = useOriginal ? document.mimeType : "image/jpeg";

  try {
    const buffer = await fs.readFile(resolveUploadPath(relativePath));
    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "private, max-age=3600",
      },
    });
  } catch {
    return NextResponse.json({ error: "File not found on disk" }, { status: 404 });
  }
}
