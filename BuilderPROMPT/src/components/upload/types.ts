import type { DocumentStatus } from "@/lib/db/schema";

export type UploadItem = {
  key: string;
  name: string;
  documentId?: string;
  status: DocumentStatus | "uploading" | "error";
  error?: string;
};
