import { NextResponse } from "next/server";
import { and, desc, eq, gte, sql } from "drizzle-orm";
import { requireSession } from "@/lib/auth/session";
import { db } from "@/lib/db";
import { documents, extractions, type DocumentStatus, documentStatuses } from "@/lib/db/schema";
import { toDocumentSummary } from "@/lib/documents/serialize";
import { runExtractionForDocument } from "@/lib/documents/pipeline";
import { processUploadedFile, sha256Hex, validateUpload } from "@/lib/upload/process";

export async function GET(request: Request) {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const url = new URL(request.url);
  const q = url.searchParams.get("q")?.trim().toLowerCase() ?? "";
  const status = url.searchParams.get("status");
  const from = url.searchParams.get("from"); // YYYY-MM-DD
  const to = url.searchParams.get("to"); // YYYY-MM-DD

  const rows = await db
    .select()
    .from(documents)
    .leftJoin(extractions, eq(extractions.documentId, documents.id))
    .where(eq(documents.userId, session.sub))
    .orderBy(desc(documents.uploadedAt));

  let items = rows.map((row) => toDocumentSummary(row.documents, row.extractions ?? undefined));

  if (status && (documentStatuses as readonly string[]).includes(status)) {
    items = items.filter((item) => item.status === (status as DocumentStatus));
  }
  if (q) {
    items = items.filter(
      (item) =>
        item.vendorName?.toLowerCase().includes(q) ||
        item.invoiceNumber?.toLowerCase().includes(q) ||
        item.originalFilename.toLowerCase().includes(q),
    );
  }
  if (from) {
    items = items.filter((item) => item.uploadedAt.slice(0, 10) >= from);
  }
  if (to) {
    items = items.filter((item) => item.uploadedAt.slice(0, 10) <= to);
  }

  const now = new Date();
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
  const [{ monthCount, monthTotal }] = await db
    .select({
      monthCount: sql<number>`count(*)`,
      monthTotal: sql<number>`coalesce(sum(json_extract(coalesce(${extractions.reviewedJson}, ${extractions.extractedJson}), '$.totals.grand_total')), 0)`,
    })
    .from(documents)
    .leftJoin(extractions, eq(extractions.documentId, documents.id))
    .where(and(eq(documents.userId, session.sub), gte(documents.uploadedAt, monthStart)));

  return NextResponse.json({
    items,
    summary: { monthCount: Number(monthCount) || 0, monthTotal: Number(monthTotal) || 0 },
  });
}

export async function POST(request: Request) {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const formData = await request.formData().catch(() => null);
  const file = formData?.get("file");
  if (!file || !(file instanceof File)) {
    return NextResponse.json({ error: "No file provided." }, { status: 400 });
  }

  const validation = validateUpload({ type: file.type, size: file.size });
  if (!validation.ok) {
    return NextResponse.json({ error: validation.error }, { status: 400 });
  }

  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  const sha256 = sha256Hex(buffer);

  const duplicate = await db
    .select({ id: documents.id })
    .from(documents)
    .where(and(eq(documents.userId, session.sub), eq(documents.sha256, sha256)))
    .limit(1);

  const processed = await processUploadedFile({ buffer, mimeType: file.type });

  const [document] = await db
    .insert(documents)
    .values({
      userId: session.sub,
      originalFilename: file.name || "document",
      storedPath: processed.storedPath,
      previewPath: processed.previewPath,
      mimeType: processed.finalMimeType,
      sizeBytes: buffer.byteLength,
      sha256,
      status: "uploaded",
    })
    .returning();

  // Fire-and-forget: this Next.js instance runs as a long-lived Node
  // process (`next start` / `next dev`), so the promise keeps running after
  // the response is sent. The client polls GET /api/documents/:id for status.
  runExtractionForDocument(document.id).catch((err) => {
    console.error(`[BillBox] unhandled extraction error for ${document.id}:`, err);
  });

  return NextResponse.json(
    { document: toDocumentSummary(document), duplicate: duplicate.length > 0 },
    { status: 201 },
  );
}
