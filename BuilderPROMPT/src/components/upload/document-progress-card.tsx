"use client";

import Link from "next/link";
import { AlertTriangle, CheckCircle2, FileText, Loader2, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { UploadItem } from "./types";

const STATUS_LABEL: Record<UploadItem["status"], string> = {
  uploading: "Uploading…",
  uploaded: "Queued…",
  processing: "Reading document…",
  extracted: "Ready to review",
  needs_review: "Ready — please double-check",
  reviewed: "Reviewed",
  failed: "Extraction failed",
  error: "Upload failed",
};

export function DocumentProgressCard({
  item,
  onDismiss,
  onRetry,
}: {
  item: UploadItem;
  onDismiss: (key: string) => void;
  onRetry: (key: string) => void;
}) {
  const isBusy = item.status === "uploading" || item.status === "uploaded" || item.status === "processing";
  const isFailed = item.status === "failed" || item.status === "error";
  const isReady = item.status === "extracted" || item.status === "needs_review";

  return (
    <div className="flex items-center gap-3 rounded-xl border bg-card p-3 shadow-sm">
      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-muted">
        {isBusy && <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />}
        {isFailed && <AlertTriangle className="h-5 w-5 text-destructive" />}
        {isReady && <CheckCircle2 className="h-5 w-5 text-primary" />}
        {item.status === "reviewed" && <FileText className="h-5 w-5 text-muted-foreground" />}
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium">{item.name}</p>
        <p
          className={cn(
            "text-xs",
            isFailed ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {item.error ?? STATUS_LABEL[item.status]}
        </p>
      </div>
      {isReady && item.documentId && (
        <Button asChild size="sm">
          <Link href={`/documents/${item.documentId}`}>Review</Link>
        </Button>
      )}
      {isFailed && (
        <Button
          size="sm"
          variant="outline"
          onClick={() => onRetry(item.key)}
          className="gap-1"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Retry
        </Button>
      )}
      {!isBusy && (
        <button
          type="button"
          onClick={() => onDismiss(item.key)}
          className="rounded-full p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
