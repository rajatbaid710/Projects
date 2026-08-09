import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { z } from "zod";
import { requireSession } from "@/lib/auth/session";
import { db } from "@/lib/db";
import { documents, extractions } from "@/lib/db/schema";
import { toDocumentDetail } from "@/lib/documents/serialize";
import { InvoiceExtractionSchema } from "@/lib/extraction/schema";
import { deleteUploadFiles } from "@/lib/upload/process";

type RouteParams = { params: Promise<{ id: string }> };

async function loadOwnedDocument(userId: string, id: string) {
  const [row] = await db
    .select()
    .from(documents)
    .leftJoin(extractions, eq(extractions.documentId, documents.id))
    .where(and(eq(documents.id, id), eq(documents.userId, userId)))
    .limit(1);
  return row;
}

export async function GET(_request: Request, { params }: RouteParams) {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const { id } = await params;
  const row = await loadOwnedDocument(session.sub, id);
  if (!row) return NextResponse.json({ error: "Document not found" }, { status: 404 });

  return NextResponse.json({
    document: toDocumentDetail(row.documents, row.extractions ?? undefined),
  });
}

const patchSchema = z.object({ reviewedJson: InvoiceExtractionSchema });

export async function PATCH(request: Request, { params }: RouteParams) {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const { id } = await params;
  const row = await loadOwnedDocument(session.sub, id);
  if (!row) return NextResponse.json({ error: "Document not found" }, { status: 404 });
  if (!row.extractions) {
    return NextResponse.json(
      { error: "This document hasn't finished extraction yet." },
      { status: 400 },
    );
  }

  const json = await request.json().catch(() => null);
  const parsed = patchSchema.safeParse(json);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid extraction data." }, { status: 400 });
  }

  await db
    .update(extractions)
    .set({ reviewedJson: parsed.data.reviewedJson, reviewedAt: new Date() })
    .where(eq(extractions.documentId, id));
  await db.update(documents).set({ status: "reviewed" }).where(eq(documents.id, id));

  const updated = await loadOwnedDocument(session.sub, id);
  return NextResponse.json({
    document: toDocumentDetail(updated!.documents, updated!.extractions ?? undefined),
  });
}

export async function DELETE(_request: Request, { params }: RouteParams) {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const { id } = await params;
  const row = await loadOwnedDocument(session.sub, id);
  if (!row) return NextResponse.json({ error: "Document not found" }, { status: 404 });

  await deleteUploadFiles([row.documents.storedPath, row.documents.previewPath]);
  await db.delete(documents).where(eq(documents.id, id));

  return NextResponse.json({ ok: true });
}
