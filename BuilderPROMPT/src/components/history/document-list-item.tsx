"use client";

import Link from "next/link";
import { FileText, Trash2 } from "lucide-react";
import { StatusBadge } from "./status-badge";
import type { DocumentSummary } from "@/lib/documents/serialize";
import { formatCurrency } from "@/components/review/field-utils";

function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // invoice_date is a date-only string (YYYY-MM-DD), parsed as UTC midnight.
  // Format in UTC too, or a negative-UTC-offset runtime would roll it back
  // to the previous day.
  return d.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function DocumentListItem({
  item,
  onDeleteClick,
}: {
  item: DocumentSummary;
  onDeleteClick: (id: string) => void;
}) {
  const isImage = item.mimeType.startsWith("image/");
  const thumbUrl = `/api/documents/${item.id}/file`;

  return (
    <div className="flex items-center gap-3 rounded-xl border bg-card p-3 shadow-sm">
      <Link href={`/documents/${item.id}`} className="flex min-w-0 flex-1 items-center gap-3">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-muted">
          {isImage ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={thumbUrl} alt="" className="h-full w-full object-cover" />
          ) : (
            <FileText className="h-5 w-5 text-muted-foreground" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="truncate text-sm font-medium">
              {item.vendorName || item.originalFilename}
            </p>
          </div>
          <p className="truncate text-xs text-muted-foreground">
            {[item.invoiceNumber, formatDate(item.invoiceDate)].filter(Boolean).join(" · ") ||
              "No details yet"}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <span className="text-sm font-semibold">
            {item.grandTotal !== null ? formatCurrency(item.grandTotal) : "—"}
          </span>
          <StatusBadge status={item.status} />
        </div>
      </Link>
      <button
        type="button"
        onClick={() => onDeleteClick(item.id)}
        className="shrink-0 rounded-full p-2 text-muted-foreground hover:bg-muted hover:text-destructive"
        aria-label="Delete document"
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );
}
