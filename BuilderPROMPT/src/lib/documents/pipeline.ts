import "server-only";

import fs from "fs/promises";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { documents, extractions } from "@/lib/db/schema";
import { extractInvoiceData } from "@/lib/extraction/extract";
import { resolveUploadPath } from "@/lib/upload/process";

const LOW_CONFIDENCE_THRESHOLD = 0.75;

/**
 * Runs (or re-runs) extraction for a document and persists the result.
 * Safe to call fire-and-forget from a Route Handler — never throws, always
 * leaves the document in a terminal status (extracted / needs_review / failed).
 */
export async function runExtractionForDocument(documentId: string): Promise<void> {
  const [document] = await db.select().from(documents).where(eq(documents.id, documentId));
  if (!document) return;

  await db
    .update(documents)
    .set({ status: "processing", errorMessage: null })
    .where(eq(documents.id, documentId));

  try {
    const sourcePath = document.previewPath ?? document.storedPath;
    const buffer = await fs.readFile(resolveUploadPath(sourcePath));
    const mimeType = document.previewPath ? "image/jpeg" : document.mimeType;

    const result = await extractInvoiceData({ buffer, mimeType });

    if (!result.ok) {
      await db
        .update(documents)
        .set({ status: "failed", errorMessage: result.error })
        .where(eq(documents.id, documentId));
      return;
    }

    const lowConfidenceFields = result.data.confidence.low_confidence_fields;
    const needsReview =
      result.data.confidence.overall < LOW_CONFIDENCE_THRESHOLD || lowConfidenceFields.length > 0;

    await db.delete(extractions).where(eq(extractions.documentId, documentId));
    await db.insert(extractions).values({
      documentId,
      model: result.model,
      extractedJson: result.data,
      overallConfidence: result.data.confidence.overall,
      lowConfidenceFields,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
      costUsd: result.costUsd,
    });

    await db
      .update(documents)
      .set({ status: needsReview ? "needs_review" : "extracted", errorMessage: null })
      .where(eq(documents.id, documentId));
  } catch (err) {
    console.error(`[BillBox] extraction pipeline error for ${documentId}:`, err);
    const message = err instanceof Error ? err.message : "Unexpected extraction error.";
    await db
      .update(documents)
      .set({ status: "failed", errorMessage: message })
      .where(eq(documents.id, documentId));
  }
}
