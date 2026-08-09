import { sql } from "drizzle-orm";
import { integer, index, real, sqliteTable, text } from "drizzle-orm/sqlite-core";
import { randomUUID } from "crypto";

export const documentStatuses = [
  "uploaded",
  "processing",
  "extracted",
  "needs_review",
  "reviewed",
  "failed",
] as const;
export type DocumentStatus = (typeof documentStatuses)[number];

export const users = sqliteTable("users", {
  id: text("id").primaryKey().$defaultFn(() => randomUUID()),
  email: text("email").notNull().unique(),
  createdAt: integer("created_at", { mode: "timestamp" })
    .notNull()
    .default(sql`(unixepoch())`),
  lastLoginAt: integer("last_login_at", { mode: "timestamp" }),
});

export const documents = sqliteTable(
  "documents",
  {
    id: text("id").primaryKey().$defaultFn(() => randomUUID()),
    userId: text("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    originalFilename: text("original_filename").notNull(),
    storedPath: text("stored_path").notNull(),
    previewPath: text("preview_path"),
    mimeType: text("mime_type").notNull(),
    sizeBytes: integer("size_bytes").notNull(),
    sha256: text("sha256").notNull(),
    status: text("status", { enum: documentStatuses }).notNull().default("uploaded"),
    errorMessage: text("error_message"),
    uploadedAt: integer("uploaded_at", { mode: "timestamp" })
      .notNull()
      .default(sql`(unixepoch())`),
  },
  (table) => [
    index("documents_user_uploaded_idx").on(table.userId, table.uploadedAt),
    index("documents_user_sha256_idx").on(table.userId, table.sha256),
  ],
);

export const extractions = sqliteTable("extractions", {
  id: text("id").primaryKey().$defaultFn(() => randomUUID()),
  documentId: text("document_id")
    .notNull()
    .unique()
    .references(() => documents.id, { onDelete: "cascade" }),
  model: text("model").notNull(),
  extractedJson: text("extracted_json", { mode: "json" }).notNull(),
  reviewedJson: text("reviewed_json", { mode: "json" }),
  overallConfidence: real("overall_confidence"),
  lowConfidenceFields: text("low_confidence_fields", { mode: "json" })
    .$type<string[]>()
    .notNull()
    .default(sql`'[]'`),
  inputTokens: integer("input_tokens").notNull().default(0),
  outputTokens: integer("output_tokens").notNull().default(0),
  costUsd: real("cost_usd").notNull().default(0),
  createdAt: integer("created_at", { mode: "timestamp" })
    .notNull()
    .default(sql`(unixepoch())`),
  reviewedAt: integer("reviewed_at", { mode: "timestamp" }),
});

export type User = typeof users.$inferSelect;
export type Document = typeof documents.$inferSelect;
export type Extraction = typeof extractions.$inferSelect;
