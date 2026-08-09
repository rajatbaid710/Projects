"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CalendarRange, Search, X } from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "@/lib/db/schema";
import type { DocumentSummary } from "@/lib/documents/serialize";
import { DocumentListItem } from "./document-list-item";
import { DeleteConfirmDialog } from "./delete-confirm-dialog";
import { formatCurrency } from "@/components/review/field-utils";

const STATUS_FILTERS: { value: DocumentStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "needs_review", label: "Needs review" },
  { value: "reviewed", label: "Reviewed" },
  { value: "processing", label: "Processing" },
  { value: "failed", label: "Failed" },
];

export function HistoryClient() {
  const [items, setItems] = useState<DocumentSummary[]>([]);
  const [summary, setSummary] = useState({ monthCount: 0, monthTotal: 0 });
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  const [status, setStatus] = useState<DocumentStatus | "all">("all");
  const [showDateFilter, setShowDateFilter] = useState(false);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const queryString = useMemo(() => {
    const params = new URLSearchParams();
    if (debouncedQ) params.set("q", debouncedQ);
    if (status !== "all") params.set("status", status);
    if (from) params.set("from", from);
    if (to) params.set("to", to);
    return params.toString();
  }, [debouncedQ, status, from, to]);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/documents${queryString ? `?${queryString}` : ""}`);
      if (!res.ok) return;
      const data = await res.json();
      setItems(data.items);
      setSummary(data.summary);
    } finally {
      setLoading(false);
    }
  }, [queryString]);

  useEffect(() => {
    // fetchItems' setState calls happen after its await, not synchronously
    // in this callback — the standard "refetch when filters change" pattern.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchItems();
  }, [fetchItems]);

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const res = await fetch(`/api/documents/${deleteTarget}`, { method: "DELETE" });
      if (res.ok) {
        toast.success("Document deleted.");
        setItems((prev) => prev.filter((it) => it.id !== deleteTarget));
      } else {
        toast.error("Could not delete this document.");
      }
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
  }

  const hasDateFilter = Boolean(from || to);

  return (
    <div className="flex flex-col gap-4 px-4 pb-8 pt-4">
      <div className="rounded-xl border bg-card p-4">
        <p className="text-xs text-muted-foreground">This month</p>
        <div className="mt-1 flex items-baseline justify-between">
          <span className="text-2xl font-semibold">{formatCurrency(summary.monthTotal)}</span>
          <span className="text-sm text-muted-foreground">
            {summary.monthCount} document{summary.monthCount === 1 ? "" : "s"}
          </span>
        </div>
      </div>

      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search vendor or invoice number…"
          className="h-11 pl-9"
        />
      </div>

      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        {STATUS_FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setStatus(f.value)}
            className={cn(
              "shrink-0 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
              status === f.value
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-background text-muted-foreground hover:text-foreground",
            )}
          >
            {f.label}
          </button>
        ))}
        <button
          onClick={() => setShowDateFilter((v) => !v)}
          className={cn(
            "flex shrink-0 items-center gap-1 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
            hasDateFilter
              ? "border-primary bg-primary text-primary-foreground"
              : "border-border bg-background text-muted-foreground hover:text-foreground",
          )}
        >
          <CalendarRange className="h-3.5 w-3.5" />
          Dates
        </button>
      </div>

      {showDateFilter && (
        <div className="flex items-center gap-2 rounded-xl border bg-card p-3">
          <div className="flex-1">
            <label className="text-[11px] text-muted-foreground">From</label>
            <Input type="date" value={from} onChange={(e) => setFrom(e.target.value)} className="h-9" />
          </div>
          <div className="flex-1">
            <label className="text-[11px] text-muted-foreground">To</label>
            <Input type="date" value={to} onChange={(e) => setTo(e.target.value)} className="h-9" />
          </div>
          {hasDateFilter && (
            <button
              onClick={() => {
                setFrom("");
                setTo("");
              }}
              className="mt-4 rounded-full p-1.5 text-muted-foreground hover:bg-muted"
              aria-label="Clear dates"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      )}

      <div className="flex flex-col gap-2">
        {loading &&
          Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-[68px] w-full rounded-xl" />
          ))}

        {!loading && items.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-16 text-center text-muted-foreground">
            <p className="text-sm">No documents match your filters yet.</p>
          </div>
        )}

        {!loading &&
          items.map((item) => (
            <DocumentListItem key={item.id} item={item} onDeleteClick={setDeleteTarget} />
          ))}
      </div>

      <DeleteConfirmDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        onConfirm={confirmDelete}
        deleting={deleting}
      />
    </div>
  );
}
