import { Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "@/lib/db/schema";

const CONFIG: Record<DocumentStatus, { label: string; className: string; spin?: boolean }> = {
  uploaded: { label: "Queued", className: "bg-muted text-muted-foreground" },
  processing: {
    label: "Processing",
    className: "bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300",
    spin: true,
  },
  extracted: {
    label: "Ready",
    className: "bg-primary/10 text-primary",
  },
  needs_review: {
    label: "Needs review",
    className: "bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300",
  },
  reviewed: {
    label: "Reviewed",
    className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300",
  },
  failed: {
    label: "Failed",
    className: "bg-destructive/10 text-destructive",
  },
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  const config = CONFIG[status];
  return (
    <Badge variant="outline" className={cn("gap-1 border-transparent font-medium", config.className)}>
      {config.spin && <Loader2 className="h-3 w-3 animate-spin" />}
      {config.label}
    </Badge>
  );
}
