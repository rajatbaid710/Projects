"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { AlertTriangle, Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { documentTypeValues, type InvoiceExtraction } from "@/lib/extraction/schema";
import { LineItemsEditor } from "./line-items-editor";
import {
  checkTotalsMismatch,
  formatCurrency,
  lowConfidenceClass,
  numberToInputValue,
  parseNumberInput,
} from "./field-utils";

const DOCUMENT_TYPE_LABELS: Record<(typeof documentTypeValues)[number], string> = {
  tax_invoice: "Tax invoice",
  bill_of_supply: "Bill of supply",
  credit_note: "Credit note",
  debit_note: "Debit note",
  receipt: "Receipt",
  delivery_challan: "Delivery challan",
  other: "Other",
};

function FieldWrap({
  label,
  path,
  lowConfidenceFields,
  children,
}: {
  label: string;
  path: string;
  lowConfidenceFields: string[];
  children: React.ReactNode;
}) {
  const flagged = lowConfidenceFields.includes(path);
  return (
    <div>
      <Label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        {flagged && <AlertTriangle className="h-3 w-3 text-amber-500" />}
      </Label>
      {children}
    </div>
  );
}

export function ReviewForm({
  documentId,
  initialData,
  lowConfidenceFields,
  alreadyReviewed,
}: {
  documentId: string;
  initialData: InvoiceExtraction;
  lowConfidenceFields: string[];
  alreadyReviewed: boolean;
}) {
  const router = useRouter();
  const [data, setData] = useState<InvoiceExtraction>(initialData);
  const [saving, setSaving] = useState(false);

  const inputCls = (path: string) => lowConfidenceClass(lowConfidenceFields, path);

  function set<K extends keyof InvoiceExtraction>(key: K, value: InvoiceExtraction[K]) {
    setData((prev) => ({ ...prev, [key]: value }));
  }
  function setVendor<K extends keyof InvoiceExtraction["vendor"]>(key: K, value: InvoiceExtraction["vendor"][K]) {
    setData((prev) => ({ ...prev, vendor: { ...prev.vendor, [key]: value } }));
  }
  function setBuyer<K extends keyof InvoiceExtraction["buyer"]>(key: K, value: InvoiceExtraction["buyer"][K]) {
    setData((prev) => ({ ...prev, buyer: { ...prev.buyer, [key]: value } }));
  }
  function setTotals<K extends keyof InvoiceExtraction["totals"]>(key: K, value: InvoiceExtraction["totals"][K]) {
    setData((prev) => ({ ...prev, totals: { ...prev.totals, [key]: value } }));
  }
  function setPayment<K extends keyof InvoiceExtraction["payment"]>(key: K, value: InvoiceExtraction["payment"][K]) {
    setData((prev) => ({ ...prev, payment: { ...prev.payment, [key]: value } }));
  }

  const mismatch = useMemo(() => checkTotalsMismatch(data.totals), [data.totals]);

  async function handleConfirm() {
    setSaving(true);
    try {
      const res = await fetch(`/api/documents/${documentId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewedJson: data }),
      });
      const body = await res.json();
      if (!res.ok) {
        toast.error(body.error ?? "Could not save this document.");
        return;
      }
      toast.success("Saved to your history.");
      router.push("/history");
    } catch {
      toast.error("Network error while saving.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 px-4 pb-8 pt-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Summary</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <FieldWrap label="Document type" path="document_type" lowConfidenceFields={lowConfidenceFields}>
              <Select value={data.document_type} onValueChange={(v) => set("document_type", v as InvoiceExtraction["document_type"])}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {documentTypeValues.map((v) => (
                    <SelectItem key={v} value={v}>
                      {DOCUMENT_TYPE_LABELS[v]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldWrap>
          </div>
          <FieldWrap label="Invoice number" path="invoice_number" lowConfidenceFields={lowConfidenceFields}>
            <Input
              value={data.invoice_number ?? ""}
              onChange={(e) => set("invoice_number", e.target.value || null)}
              className={inputCls("invoice_number")}
            />
          </FieldWrap>
          <FieldWrap label="Invoice date" path="invoice_date" lowConfidenceFields={lowConfidenceFields}>
            <Input
              type="date"
              value={data.invoice_date ?? ""}
              onChange={(e) => set("invoice_date", e.target.value || null)}
              className={inputCls("invoice_date")}
            />
          </FieldWrap>
          <FieldWrap label="Due date" path="due_date" lowConfidenceFields={lowConfidenceFields}>
            <Input
              type="date"
              value={data.due_date ?? ""}
              onChange={(e) => set("due_date", e.target.value || null)}
              className={inputCls("due_date")}
            />
          </FieldWrap>
          <label className="col-span-2 mt-1 flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={data.is_handwritten}
              onChange={(e) => set("is_handwritten", e.target.checked)}
              className="h-4 w-4 rounded border-input accent-primary"
            />
            Handwritten document
          </label>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Vendor</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <FieldWrap label="Name" path="vendor.name" lowConfidenceFields={lowConfidenceFields}>
              <Input value={data.vendor.name ?? ""} onChange={(e) => setVendor("name", e.target.value || null)} className={inputCls("vendor.name")} />
            </FieldWrap>
          </div>
          <FieldWrap label="GSTIN" path="vendor.gstin" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.vendor.gstin ?? ""} onChange={(e) => setVendor("gstin", e.target.value.toUpperCase() || null)} className={inputCls("vendor.gstin")} />
          </FieldWrap>
          <FieldWrap label="State" path="vendor.state" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.vendor.state ?? ""} onChange={(e) => setVendor("state", e.target.value || null)} className={inputCls("vendor.state")} />
          </FieldWrap>
          <FieldWrap label="Phone" path="vendor.phone" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.vendor.phone ?? ""} onChange={(e) => setVendor("phone", e.target.value || null)} className={inputCls("vendor.phone")} />
          </FieldWrap>
          <FieldWrap label="Email" path="vendor.email" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.vendor.email ?? ""} onChange={(e) => setVendor("email", e.target.value || null)} className={inputCls("vendor.email")} />
          </FieldWrap>
          <div className="col-span-2">
            <FieldWrap label="Address" path="vendor.address" lowConfidenceFields={lowConfidenceFields}>
              <Textarea value={data.vendor.address ?? ""} onChange={(e) => setVendor("address", e.target.value || null)} className={inputCls("vendor.address")} rows={2} />
            </FieldWrap>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Buyer</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <FieldWrap label="Name" path="buyer.name" lowConfidenceFields={lowConfidenceFields}>
              <Input value={data.buyer.name ?? ""} onChange={(e) => setBuyer("name", e.target.value || null)} className={inputCls("buyer.name")} />
            </FieldWrap>
          </div>
          <FieldWrap label="GSTIN" path="buyer.gstin" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.buyer.gstin ?? ""} onChange={(e) => setBuyer("gstin", e.target.value.toUpperCase() || null)} className={inputCls("buyer.gstin")} />
          </FieldWrap>
          <FieldWrap label="State" path="buyer.state" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.buyer.state ?? ""} onChange={(e) => setBuyer("state", e.target.value || null)} className={inputCls("buyer.state")} />
          </FieldWrap>
          <div className="col-span-2">
            <FieldWrap label="Address" path="buyer.address" lowConfidenceFields={lowConfidenceFields}>
              <Textarea value={data.buyer.address ?? ""} onChange={(e) => setBuyer("address", e.target.value || null)} className={inputCls("buyer.address")} rows={2} />
            </FieldWrap>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">GST details</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3">
          <FieldWrap label="Place of supply" path="place_of_supply" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.place_of_supply ?? ""} onChange={(e) => set("place_of_supply", e.target.value || null)} className={inputCls("place_of_supply")} />
          </FieldWrap>
          <FieldWrap label="IRN" path="irn" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.irn ?? ""} onChange={(e) => set("irn", e.target.value || null)} className={inputCls("irn")} />
          </FieldWrap>
          <label className="col-span-2 flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={data.reverse_charge ?? false}
              onChange={(e) => set("reverse_charge", e.target.checked)}
              className="h-4 w-4 rounded border-input accent-primary"
            />
            Reverse charge applicable
          </label>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Line items</CardTitle>
        </CardHeader>
        <CardContent>
          <LineItemsEditor items={data.line_items} onChange={(items) => set("line_items", items)} />
        </CardContent>
      </Card>

      {mismatch !== null && (
        <Alert variant="destructive" className="border-amber-400 bg-amber-50 text-amber-900 dark:bg-amber-950/30 dark:text-amber-200 [&>svg]:text-amber-600">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Totals don&apos;t quite add up</AlertTitle>
          <AlertDescription>
            Line items + tax − discount ≠ grand total (off by {formatCurrency(Math.abs(mismatch), data.currency)}). Worth a quick check before confirming.
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Totals</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3">
          <FieldWrap label="Taxable value" path="totals.taxable_value" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.taxable_value)} onChange={(e) => setTotals("taxable_value", parseNumberInput(e.target.value))} className={inputCls("totals.taxable_value")} />
          </FieldWrap>
          <FieldWrap label="Discount" path="totals.discount_total" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.discount_total)} onChange={(e) => setTotals("discount_total", parseNumberInput(e.target.value))} className={inputCls("totals.discount_total")} />
          </FieldWrap>
          <FieldWrap label="CGST" path="totals.cgst_total" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.cgst_total)} onChange={(e) => setTotals("cgst_total", parseNumberInput(e.target.value))} className={inputCls("totals.cgst_total")} />
          </FieldWrap>
          <FieldWrap label="SGST" path="totals.sgst_total" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.sgst_total)} onChange={(e) => setTotals("sgst_total", parseNumberInput(e.target.value))} className={inputCls("totals.sgst_total")} />
          </FieldWrap>
          <FieldWrap label="IGST" path="totals.igst_total" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.igst_total)} onChange={(e) => setTotals("igst_total", parseNumberInput(e.target.value))} className={inputCls("totals.igst_total")} />
          </FieldWrap>
          <FieldWrap label="Cess" path="totals.cess_total" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.cess_total)} onChange={(e) => setTotals("cess_total", parseNumberInput(e.target.value))} className={inputCls("totals.cess_total")} />
          </FieldWrap>
          <FieldWrap label="Round off" path="totals.round_off" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.round_off)} onChange={(e) => setTotals("round_off", parseNumberInput(e.target.value))} className={inputCls("totals.round_off")} />
          </FieldWrap>
          <FieldWrap label="Grand total" path="totals.grand_total" lowConfidenceFields={lowConfidenceFields}>
            <Input inputMode="decimal" value={numberToInputValue(data.totals.grand_total)} onChange={(e) => setTotals("grand_total", parseNumberInput(e.target.value))} className={`font-semibold ${inputCls("totals.grand_total")}`} />
          </FieldWrap>
          <div className="col-span-2">
            <FieldWrap label="Amount in words" path="totals.amount_in_words" lowConfidenceFields={lowConfidenceFields}>
              <Textarea value={data.totals.amount_in_words ?? ""} onChange={(e) => setTotals("amount_in_words", e.target.value || null)} rows={2} className={inputCls("totals.amount_in_words")} />
            </FieldWrap>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Payment</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3">
          <FieldWrap label="Mode" path="payment.mode" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.payment.mode ?? ""} onChange={(e) => setPayment("mode", e.target.value || null)} className={inputCls("payment.mode")} />
          </FieldWrap>
          <FieldWrap label="Bank name" path="payment.bank_name" lowConfidenceFields={lowConfidenceFields}>
            <Input value={data.payment.bank_name ?? ""} onChange={(e) => setPayment("bank_name", e.target.value || null)} className={inputCls("payment.bank_name")} />
          </FieldWrap>
          <div className="col-span-2">
            <FieldWrap label="UPI ID" path="payment.upi_id" lowConfidenceFields={lowConfidenceFields}>
              <Input value={data.payment.upi_id ?? ""} onChange={(e) => setPayment("upi_id", e.target.value || null)} className={inputCls("payment.upi_id")} />
            </FieldWrap>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Notes</CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea
            value={data.notes ?? ""}
            onChange={(e) => set("notes", e.target.value || null)}
            rows={3}
            placeholder="Anything unusual about this document…"
          />
        </CardContent>
      </Card>

      <Button size="lg" className="h-14 w-full text-base" disabled={saving} onClick={handleConfirm}>
        {saving ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" /> Saving…
          </>
        ) : alreadyReviewed ? (
          "Save changes"
        ) : (
          "Confirm & save"
        )}
      </Button>
    </div>
  );
}
