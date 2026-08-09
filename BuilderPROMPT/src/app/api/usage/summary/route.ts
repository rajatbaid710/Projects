import { NextResponse } from "next/server";
import { and, eq, gte, sql } from "drizzle-orm";
import { requireSession } from "@/lib/auth/session";
import { db } from "@/lib/db";
import { documents, extractions } from "@/lib/db/schema";

export async function GET() {
  const session = await requireSession();
  if (!session) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  const now = new Date();
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);

  const [totals] = await db
    .select({
      totalCostUsd: sql<number>`coalesce(sum(${extractions.costUsd}), 0)`,
      totalDocuments: sql<number>`count(*)`,
    })
    .from(extractions)
    .innerJoin(documents, eq(documents.id, extractions.documentId))
    .where(eq(documents.userId, session.sub));

  const [monthTotals] = await db
    .select({ monthCostUsd: sql<number>`coalesce(sum(${extractions.costUsd}), 0)` })
    .from(extractions)
    .innerJoin(documents, eq(documents.id, extractions.documentId))
    .where(and(eq(documents.userId, session.sub), gte(documents.uploadedAt, monthStart)));

  return NextResponse.json({
    totalCostUsd: Number(totals?.totalCostUsd ?? 0),
    monthCostUsd: Number(monthTotals?.monthCostUsd ?? 0),
    totalDocuments: Number(totals?.totalDocuments ?? 0),
    model: process.env.EXTRACTION_MODEL?.trim() || "claude-sonnet-5",
  });
}
