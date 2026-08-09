"use client";

import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { LineItem } from "@/lib/extraction/schema";
import { numberToInputValue, parseNumberInput } from "./field-utils";

const EMPTY_LINE_ITEM: LineItem = {
  description: "",
  hsn_sac: null,
  quantity: null,
  unit: null,
  rate: null,
  discount: null,
  taxable_value: null,
  gst_rate: null,
  cgst: null,
  sgst: null,
  igst: null,
  cess: null,
  total: null,
};

function MiniField({
  label,
  value,
  onChange,
  numeric,
  className,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  numeric?: boolean;
  className?: string;
}) {
  return (
    <div className={className}>
      <Label className="text-[11px] font-normal text-muted-foreground">{label}</Label>
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        inputMode={numeric ? "decimal" : undefined}
        className="h-9 text-sm"
      />
    </div>
  );
}

export function LineItemsEditor({
  items,
  onChange,
}: {
  items: LineItem[];
  onChange: (items: LineItem[]) => void;
}) {
  function updateItem(index: number, patch: Partial<LineItem>) {
    onChange(items.map((it, i) => (i === index ? { ...it, ...patch } : it)));
  }

  function removeItem(index: number) {
    onChange(items.filter((_, i) => i !== index));
  }

  function addItem() {
    onChange([...items, { ...EMPTY_LINE_ITEM }]);
  }

  return (
    <div className="flex flex-col gap-3">
      {items.length === 0 && (
        <p className="rounded-lg border border-dashed p-4 text-center text-sm text-muted-foreground">
          No line items yet.
        </p>
      )}
      {items.map((item, index) => (
        <div key={index} className="rounded-xl border bg-card p-3">
          <div className="mb-2 flex items-start gap-2">
            <div className="flex-1">
              <Label className="text-[11px] font-normal text-muted-foreground">
                Description
              </Label>
              <Input
                value={item.description}
                onChange={(e) => updateItem(index, { description: e.target.value })}
                className="h-9 text-sm"
              />
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="mt-4 h-9 w-9 shrink-0 text-muted-foreground hover:text-destructive"
              onClick={() => removeItem(index)}
              aria-label="Remove line item"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <MiniField
              label="HSN / SAC"
              value={item.hsn_sac ?? ""}
              onChange={(v) => updateItem(index, { hsn_sac: v || null })}
            />
            <MiniField
              label="Unit"
              value={item.unit ?? ""}
              onChange={(v) => updateItem(index, { unit: v || null })}
            />
          </div>

          <div className="mt-2 grid grid-cols-3 gap-2">
            <MiniField
              label="Qty"
              numeric
              value={numberToInputValue(item.quantity)}
              onChange={(v) => updateItem(index, { quantity: parseNumberInput(v) })}
            />
            <MiniField
              label="Rate"
              numeric
              value={numberToInputValue(item.rate)}
              onChange={(v) => updateItem(index, { rate: parseNumberInput(v) })}
            />
            <MiniField
              label="Discount"
              numeric
              value={numberToInputValue(item.discount)}
              onChange={(v) => updateItem(index, { discount: parseNumberInput(v) })}
            />
          </div>

          <div className="mt-2 grid grid-cols-2 gap-2">
            <MiniField
              label="GST %"
              numeric
              value={numberToInputValue(item.gst_rate)}
              onChange={(v) => updateItem(index, { gst_rate: parseNumberInput(v) })}
            />
            <MiniField
              label="Taxable value"
              numeric
              value={numberToInputValue(item.taxable_value)}
              onChange={(v) => updateItem(index, { taxable_value: parseNumberInput(v) })}
            />
          </div>

          <div className="mt-2 grid grid-cols-4 gap-2">
            <MiniField
              label="CGST"
              numeric
              value={numberToInputValue(item.cgst)}
              onChange={(v) => updateItem(index, { cgst: parseNumberInput(v) })}
            />
            <MiniField
              label="SGST"
              numeric
              value={numberToInputValue(item.sgst)}
              onChange={(v) => updateItem(index, { sgst: parseNumberInput(v) })}
            />
            <MiniField
              label="IGST"
              numeric
              value={numberToInputValue(item.igst)}
              onChange={(v) => updateItem(index, { igst: parseNumberInput(v) })}
            />
            <MiniField
              label="Cess"
              numeric
              value={numberToInputValue(item.cess)}
              onChange={(v) => updateItem(index, { cess: parseNumberInput(v) })}
            />
          </div>

          <div className="mt-2">
            <MiniField
              label="Line total"
              numeric
              value={numberToInputValue(item.total)}
              onChange={(v) => updateItem(index, { total: parseNumberInput(v) })}
              className="[&_input]:font-semibold"
            />
          </div>
        </div>
      ))}

      <Button type="button" variant="outline" onClick={addItem} className="gap-2">
        <Plus className="h-4 w-4" />
        Add line item
      </Button>
    </div>
  );
}
