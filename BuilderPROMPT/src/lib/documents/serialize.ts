import type { Document, DocumentStatus, Extraction } from "@/lib/db/schema";
import type { InvoiceExtraction } from "@/lib/extraction/schema";

export type DocumentSummary = {
  id: string;
  status: DocumentStatus;
  uploadedAt: string;
  originalFilename: string;
  mimeType: string;
  errorMessage: string | null;
  vendorName: string | null;
  invoiceNumber: string | null;
  invoiceDate: string | null;
  grandTotal: number | null;
  isHandwritten: boolean;
};

export type DocumentDetail = DocumentSummary & {
  extraction: InvoiceExtraction | null;
  rawExtraction: InvoiceExtraction | null;
  lowConfidenceFields: string[];
  overallConfidence: number | null;
  reviewedAt: string | null;
  model: string | null;
  costUsd: number | null;
};

function effectiveData(extraction: Extraction | undefined): InvoiceExtraction | null {
  if (!extraction) return null;
  return (extraction.reviewedJson ?? extraction.extractedJson) as InvoiceExtraction;
}

export function toDocumentSummary(document: Document, extraction?: Extraction): DocumentSummary {
  const data = effectiveData(extraction);
  return {
    id: document.id,
    status: document.status,
    uploadedAt: document.uploadedAt.toISOString(),
    originalFilename: document.originalFilename,
    mimeType: document.mimeType,
    errorMessage: document.errorMessage,
    vendorName: data?.vendor?.name ?? null,
    invoiceNumber: data?.invoice_number ?? null,
    invoiceDate: data?.invoice_date ?? null,
    grandTotal: data?.totals?.grand_total ?? null,
    isHandwritten: data?.is_handwritten ?? false,
  };
}

export function toDocumentDetail(document: Document, extraction?: Extraction): DocumentDetail {
  return {
    ...toDocumentSummary(document, extraction),
    extraction: effectiveData(extraction),
    rawExtraction: (extraction?.extractedJson as InvoiceExtraction | undefined) ?? null,
    lowConfidenceFields: extraction?.lowConfidenceFields ?? [],
    overallConfidence: extraction?.overallConfidence ?? null,
    reviewedAt: extraction?.reviewedAt ? extraction.reviewedAt.toISOString() : null,
    model: extraction?.model ?? null,
    costUsd: extraction?.costUsd ?? null,
  };
}
