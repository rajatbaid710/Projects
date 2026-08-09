import type { InvoiceExtraction } from "@/lib/extraction/schema";

export function parseNumberInput(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

export function numberToInputValue(n: number | null | undefined): string {
  return n === null || n === undefined ? "" : String(n);
}

export function isLowConfidence(lowConfidenceFields: string[], path: string): boolean {
  return lowConfidenceFields.includes(path);
}

export function lowConfidenceClass(lowConfidenceFields: string[], path: string): string {
  return isLowConfidence(lowConfidenceFields, path)
    ? "border-amber-400 bg-amber-50/60 focus-visible:ring-amber-400/40 dark:bg-amber-950/20 dark:border-amber-600"
    : "";
}

const ROUNDING_TOLERANCE = 1;

export function checkTotalsMismatch(totals: InvoiceExtraction["totals"]): number | null {
  if (totals.grand_total === null) return null;
  const computed =
    (totals.taxable_value ?? 0) +
    (totals.cgst_total ?? 0) +
    (totals.sgst_total ?? 0) +
    (totals.igst_total ?? 0) +
    (totals.cess_total ?? 0) -
    (totals.discount_total ?? 0) +
    (totals.round_off ?? 0);
  const diff = Math.round((computed - totals.grand_total) * 100) / 100;
  return Math.abs(diff) > ROUNDING_TOLERANCE ? diff : null;
}

export function formatCurrency(amount: number | null, currency = "INR"): string {
  if (amount === null) return "—";
  try {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(amount);
  } catch {
    return amount.toFixed(2);
  }
}
