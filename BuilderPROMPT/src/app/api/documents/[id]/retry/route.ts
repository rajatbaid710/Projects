import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { requireSession } from "@/lib/auth/session";
import { db } from "@/lib/db";
import { documents } from "@/lib/db/schema";
import { toDocumentSummary } from "@/lib/documents/serialize";
import { runExtractionForDocument } from "@/lib/documents/pipeline";

type RouteParams = { params: Promise<{ id: string }> };

export async function POST(_request: Request, { params }: RouteParams) {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const { id } = await params;
  const [document] = await db
    .select()
    .from(documents)
    .where(and(eq(documents.id, id), eq(documents.userId, session.sub)))
    .limit(1);

  if (!document) return NextResponse.json({ error: "Document not found" }, { status: 404 });
  if (document.status !== "failed") {
    return NextResponse.json({ error: "Only failed documents can be retried." }, { status: 400 });
  }

  runExtractionForDocument(document.id).catch((err) => {
    console.error(`[BillBox] unhandled retry extraction error for ${document.id}:`, err);
  });

  return NextResponse.json({ document: toDocumentSummary(document) });
}
