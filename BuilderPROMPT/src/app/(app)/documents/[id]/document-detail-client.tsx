"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { AlertTriangle, ArrowLeft, Loader2, RotateCcw, Trash2 } from "lucide-react";
import { AppHeader } from "@/components/nav/app-header";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ReviewForm } from "@/components/review/review-form";
import type { DocumentDetail } from "@/lib/documents/serialize";

const TERMINAL_STATUSES = new Set(["extracted", "needs_review", "reviewed", "failed"]);

const STATUS_SUBTITLE: Record<string, string> = {
  uploaded: "Queued…",
  processing: "Reading document…",
  extracted: "Ready to review",
  needs_review: "Please double-check the highlighted fields",
  reviewed: "Reviewed",
  failed: "Extraction failed",
};

export function DocumentDetailClient({ id }: { id: string }) {
  const router = useRouter();
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [missing, setMissing] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const fetchDoc = useCallback(async () => {
    const res = await fetch(`/api/documents/${id}`);
    if (res.status === 404) {
      setMissing(true);
      setLoading(false);
      return;
    }
    if (res.ok) {
      const data = await res.json();
      setDoc(data.document);
    }
    setLoading(false);
  }, [id]);

  useEffect(() => {
    // Initial load — fetchDoc's own setState calls happen after the await,
    // not synchronously in this callback, so this is the standard
    // fetch-on-mount pattern (see react.dev/learn/synchronizing-with-effects).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchDoc();
  }, [fetchDoc]);

  useEffect(() => {
    if (!doc || TERMINAL_STATUSES.has(doc.status)) return;
    const interval = setInterval(fetchDoc, 2000);
    return () => clearInterval(interval);
  }, [doc, fetchDoc]);

  async function handleRetry() {
    setDoc((prev) => (prev ? { ...prev, status: "processing" } : prev));
    await fetch(`/api/documents/${id}/retry`, { method: "POST" });
    fetchDoc();
  }

  async function handleDelete() {
    setDeleting(true);
    try {
      const res = await fetch(`/api/documents/${id}`, { method: "DELETE" });
      if (res.ok) {
        toast.success("Document deleted.");
        router.push("/history");
        return;
      }
      toast.error("Could not delete this document.");
    } finally {
      setDeleting(false);
      setDeleteOpen(false);
    }
  }

  if (missing) {
    return (
      <div className="flex h-[70vh] flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-muted-foreground">This document doesn&apos;t exist or isn&apos;t yours.</p>
        <Button onClick={() => router.push("/history")}>Back to history</Button>
      </div>
    );
  }

  const fileUrl = `/api/documents/${id}/file`;

  return (
    <div className="flex flex-col">
      <AppHeader
        title={doc?.vendorName || doc?.originalFilename || "Document"}
        subtitle={doc ? STATUS_SUBTITLE[doc.status] : undefined}
        leading={
          <button
            onClick={() => router.push("/history")}
            className="-ml-1.5 rounded-full p-2 hover:bg-muted"
            aria-label="Back to history"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
        }
        action={
          doc && (
            <button
              onClick={() => setDeleteOpen(true)}
              className="-mr-1.5 rounded-full p-2 text-muted-foreground hover:bg-muted hover:text-destructive"
              aria-label="Delete document"
            >
              <Trash2 className="h-5 w-5" />
            </button>
          )
        }
      />

      {loading && (
        <div className="flex h-[50vh] items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {doc && (doc.status === "uploaded" || doc.status === "processing") && (
        <div className="flex h-[50vh] flex-col items-center justify-center gap-3 px-6 text-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">
            Reading the document and pulling out the details…
          </p>
        </div>
      )}

      {doc && doc.status === "failed" && (
        <div className="flex flex-col items-center gap-4 px-6 py-12 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-destructive/10">
            <AlertTriangle className="h-6 w-6 text-destructive" />
          </div>
          <div>
            <p className="font-medium">Extraction failed</p>
            <p className="mt-1 text-sm text-muted-foreground">
              {doc.errorMessage ?? "Something went wrong reading this document."}
            </p>
          </div>
          <Button onClick={handleRetry} className="gap-2">
            <RotateCcw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      )}

      {doc && doc.extraction && (
        <>
          <details open className="mx-4 mt-4 overflow-hidden rounded-xl border bg-card">
            <summary className="cursor-pointer select-none px-3 py-2.5 text-sm font-medium text-muted-foreground">
              Document preview
            </summary>
            <div className="border-t p-2">
              {doc.mimeType === "application/pdf" ? (
                <iframe src={fileUrl} className="h-[50vh] w-full rounded-lg" title="Document preview" />
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={fileUrl}
                  alt="Document preview"
                  className="mx-auto max-h-[50vh] w-auto rounded-lg bg-muted object-contain"
                />
              )}
            </div>
          </details>

          <ReviewForm
            documentId={id}
            initialData={doc.extraction}
            lowConfidenceFields={doc.lowConfidenceFields}
            alreadyReviewed={doc.status === "reviewed"}
          />
        </>
      )}

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this document?</DialogTitle>
            <DialogDescription>
              This permanently removes the file and its extracted data. This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
              {deleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
