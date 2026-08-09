import { z } from "zod";

export const documentTypeValues = [
  "tax_invoice",
  "bill_of_supply",
  "credit_note",
  "debit_note",
  "receipt",
  "delivery_challan",
  "other",
] as const;

const partySchema = z.object({
  name: z.string().nullable(),
  gstin: z.string().nullable(),
  address: z.string().nullable(),
  state: z.string().nullable(),
});

const lineItemSchema = z.object({
  description: z.string(),
  hsn_sac: z.string().nullable(),
  quantity: z.number().nullable(),
  unit: z.string().nullable(),
  rate: z.number().nullable(),
  discount: z.number().nullable(),
  taxable_value: z.number().nullable(),
  gst_rate: z.number().nullable(),
  cgst: z.number().nullable(),
  sgst: z.number().nullable(),
  igst: z.number().nullable(),
  cess: z.number().nullable(),
  total: z.number().nullable(),
});

export const InvoiceExtractionSchema = z.object({
  document_type: z.enum(documentTypeValues),
  is_handwritten: z.boolean(),
  invoice_number: z.string().nullable(),
  invoice_date: z.string().nullable(),
  due_date: z.string().nullable(),
  vendor: z.object({
    name: z.string().nullable(),
    gstin: z.string().nullable(),
    address: z.string().nullable(),
    state: z.string().nullable(),
    phone: z.string().nullable(),
    email: z.string().nullable(),
  }),
  buyer: partySchema,
  place_of_supply: z.string().nullable(),
  reverse_charge: z.boolean().nullable(),
  irn: z.string().nullable(),
  line_items: z.array(lineItemSchema),
  totals: z.object({
    taxable_value: z.number().nullable(),
    cgst_total: z.number().nullable(),
    sgst_total: z.number().nullable(),
    igst_total: z.number().nullable(),
    cess_total: z.number().nullable(),
    discount_total: z.number().nullable(),
    round_off: z.number().nullable(),
    grand_total: z.number().nullable(),
    amount_in_words: z.string().nullable(),
  }),
  currency: z.string(),
  payment: z.object({
    mode: z.string().nullable(),
    bank_name: z.string().nullable(),
    upi_id: z.string().nullable(),
  }),
  notes: z.string().nullable(),
  confidence: z.object({
    overall: z.number(),
    low_confidence_fields: z.array(z.string()),
  }),
});

export type InvoiceExtraction = z.infer<typeof InvoiceExtractionSchema>;
export type LineItem = z.infer<typeof lineItemSchema>;

export const EMPTY_EXTRACTION: InvoiceExtraction = {
  document_type: "other",
  is_handwritten: false,
  invoice_number: null,
  invoice_date: null,
  due_date: null,
  vendor: { name: null, gstin: null, address: null, state: null, phone: null, email: null },
  buyer: { name: null, gstin: null, address: null, state: null },
  place_of_supply: null,
  reverse_charge: null,
  irn: null,
  line_items: [],
  totals: {
    taxable_value: null,
    cgst_total: null,
    sgst_total: null,
    igst_total: null,
    cess_total: null,
    discount_total: null,
    round_off: null,
    grand_total: null,
    amount_in_words: null,
  },
  currency: "INR",
  payment: { mode: null, bank_name: null, upi_id: null },
  notes: null,
  confidence: { overall: 0, low_confidence_fields: [] },
};
