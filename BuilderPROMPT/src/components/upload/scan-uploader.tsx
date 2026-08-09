"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Camera, FolderOpen } from "lucide-react";
import { toast } from "sonner";
import { DocumentProgressCard } from "./document-progress-card";
import type { UploadItem } from "./types";

const TERMINAL_STATUSES = new Set(["extracted", "needs_review", "reviewed", "failed"]);
const MAX_SIZE_BYTES = 20 * 1024 * 1024;

function makeKey() {
  return Math.random().toString(36).slice(2);
}

export function ScanUploader() {
  const router = useRouter();
  const [items, setItems] = useState<UploadItem[]>([]);
  const navigatedRef = useRef(false);

  const updateItem = useCallback((key: string, patch: Partial<UploadItem>) => {
    setItems((prev) => prev.map((it) => (it.key === key ? { ...it, ...patch } : it)));
  }, []);

  const uploadFile = useCallback(
    async (key: string, file: File) => {
      if (file.size > MAX_SIZE_BYTES) {
        updateItem(key, { status: "error", error: "File is too large. Maximum size is 20 MB." });
        return;
      }
      updateItem(key, { status: "uploading" });
      try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/api/documents", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) {
          updateItem(key, { status: "error", error: data.error ?? "Upload failed." });
          return;
        }
        if (data.duplicate) {
          toast.warning(`"${file.name}" looks like a document you already uploaded.`);
        }
        updateItem(key, { status: data.document.status, documentId: data.document.id });
      } catch {
        updateItem(key, { status: "error", error: "Network error during upload." });
      }
    },
    [updateItem],
  );

  const handleFiles = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0) return;
      navigatedRef.current = false;
      const files = Array.from(fileList);
      const newItems: UploadItem[] = files.map((f) => ({
        key: makeKey(),
        name: f.name,
        status: "uploading",
      }));
      setItems((prev) => [...newItems, ...prev]);

      // Sequential, matching the low-ops-cost design: one extraction call at a time.
      for (let i = 0; i < files.length; i++) {
        await uploadFile(newItems[i].key, files[i]);
      }
    },
    [uploadFile],
  );

  const retryItem = useCallback(
    async (key: string) => {
      const item = items.find((it) => it.key === key);
      if (!item?.documentId) return;
      updateItem(key, { status: "processing", error: undefined });
      try {
        const res = await fetch(`/api/documents/${item.documentId}/retry`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          updateItem(key, { status: "failed", error: data.error ?? "Retry failed." });
        }
      } catch {
        updateItem(key, { status: "failed", error: "Network error during retry." });
      }
    },
    [items, updateItem],
  );

  const dismissItem = useCallback((key: string) => {
    setItems((prev) => prev.filter((it) => it.key !== key));
  }, []);

  // Poll status for anything still in flight.
  useEffect(() => {
    const pending = items.filter(
      (it) => it.documentId && !TERMINAL_STATUSES.has(it.status) && it.status !== "error",
    );
    if (pending.length === 0) return;

    const interval = setInterval(async () => {
      await Promise.all(
        pending.map(async (it) => {
          try {
            const res = await fetch(`/api/documents/${it.documentId}`);
            if (!res.ok) return;
            const data = await res.json();
            updateItem(it.key, { status: data.document.status, error: data.document.errorMessage ?? undefined });
          } catch {
            // transient network hiccup — try again next tick
          }
        }),
      );
    }, 2000);

    return () => clearInterval(interval);
  }, [items, updateItem]);

  // Auto-navigate straight to review when there's exactly one document and it's ready.
  useEffect(() => {
    if (items.length !== 1 || navigatedRef.current) return;
    const [only] = items;
    if ((only.status === "extracted" || only.status === "needs_review") && only.documentId) {
      navigatedRef.current = true;
      router.push(`/documents/${only.documentId}`);
    }
  }, [items, router]);

  return (
    <div className="flex flex-col gap-6 px-4 pb-6 pt-6">
      <div className="grid grid-cols-2 gap-3">
        {/*
          Native <label htmlFor> association, not a button + ref.click().
          A JS-forwarded click (button onClick -> inputRef.current.click())
          silently does nothing if it fires before hydration attaches the
          handler, and some mobile browsers are also reluctant to open the
          file picker via .click() on a display:none input. A label/input
          pair opens the picker natively, with no JS involved at all.
        */}
        <label
          htmlFor="camera-file-input"
          className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl bg-primary py-8 text-primary-foreground shadow-sm active:scale-[0.98] transition-transform"
        >
          <Camera className="h-7 w-7" />
          <span className="text-sm font-semibold">Scan invoice</span>
        </label>
        <label
          htmlFor="gallery-file-input"
          className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed py-8 text-foreground active:scale-[0.98] transition-transform"
        >
          <FolderOpen className="h-7 w-7 text-muted-foreground" />
          <span className="text-sm font-semibold">Choose files</span>
        </label>
      </div>

      <input
        id="camera-file-input"
        type="file"
        accept="image/*,application/pdf"
        capture="environment"
        className="sr-only"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <input
        id="gallery-file-input"
        type="file"
        accept="image/*,application/pdf"
        multiple
        className="sr-only"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {items.length > 0 && (
        <div className="flex flex-col gap-2">
          <h2 className="px-1 text-sm font-medium text-muted-foreground">This session</h2>
          {items.map((item) => (
            <DocumentProgressCard
              key={item.key}
              item={item}
              onDismiss={dismissItem}
              onRetry={retryItem}
            />
          ))}
        </div>
      )}

      {items.length === 0 && (
        <div className="mt-8 flex flex-col items-center gap-2 text-center text-muted-foreground">
          <Camera className="h-8 w-8 opacity-40" />
          <p className="max-w-[22rem] text-sm">
            Photograph a bill or invoice — printed or handwritten — and we&apos;ll pull out the
            key details automatically.
          </p>
        </div>
      )}
    </div>
  );
}
