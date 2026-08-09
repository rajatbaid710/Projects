import "server-only";

import { createHash, randomUUID } from "crypto";
import fs from "fs/promises";
import path from "path";
import sharp from "sharp";
import heicConvert from "heic-convert";

export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024; // 20 MB
const PREVIEW_LONG_EDGE = 1568;
const PREVIEW_QUALITY = 85;

const HEIC_MIME_TYPES = new Set(["image/heic", "image/heif"]);
const IMAGE_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
export const ACCEPTED_MIME_TYPES = new Set([
  ...IMAGE_MIME_TYPES,
  ...HEIC_MIME_TYPES,
  "application/pdf",
]);

export const UPLOAD_DIR = path.join(process.cwd(), "data", "uploads");

export function validateUpload(file: {
  type: string;
  size: number;
}): { ok: true } | { ok: false; error: string } {
  if (file.size <= 0) return { ok: false, error: "The file is empty." };
  if (file.size > MAX_UPLOAD_BYTES) {
    return { ok: false, error: "File is too large. Maximum size is 20 MB." };
  }
  if (!ACCEPTED_MIME_TYPES.has(file.type)) {
    return {
      ok: false,
      error: "Unsupported file type. Upload a PDF, JPEG, PNG, WebP, or HEIC photo.",
    };
  }
  return { ok: true };
}

export function sha256Hex(buffer: Buffer): string {
  return createHash("sha256").update(buffer).digest("hex");
}

async function ensureUploadDir() {
  await fs.mkdir(UPLOAD_DIR, { recursive: true });
}

export function resolveUploadPath(relativePath: string): string {
  return path.join(UPLOAD_DIR, relativePath);
}

type ProcessResult = {
  storedPath: string; // relative to UPLOAD_DIR — the original, full-resolution file
  previewPath: string | null; // relative to UPLOAD_DIR — downscaled JPEG for display + AI, null for PDFs
  finalMimeType: string;
};

/**
 * Normalizes the uploaded bytes (HEIC -> JPEG), writes the original to disk,
 * and — for images — writes a downscaled preview used for both on-screen
 * display and as the payload sent to the extraction agent (keeps AI cost low).
 */
export async function processUploadedFile(input: {
  buffer: Buffer;
  mimeType: string;
}): Promise<ProcessResult> {
  await ensureUploadDir();
  const id = randomUUID();

  let buffer = input.buffer;
  let mimeType = input.mimeType;

  if (HEIC_MIME_TYPES.has(mimeType)) {
    const converted = await heicConvert({
      buffer: new Uint8Array(buffer),
      format: "JPEG",
      quality: 0.92,
    });
    buffer = Buffer.from(converted);
    mimeType = "image/jpeg";
  }

  if (mimeType === "application/pdf") {
    const storedPath = `${id}.pdf`;
    await fs.writeFile(resolveUploadPath(storedPath), buffer);
    return { storedPath, previewPath: null, finalMimeType: mimeType };
  }

  const ext = mimeType === "image/png" ? "png" : mimeType === "image/webp" ? "webp" : "jpg";
  const storedPath = `${id}-original.${ext}`;
  await fs.writeFile(resolveUploadPath(storedPath), buffer);

  const previewPath = `${id}-preview.jpg`;
  const previewBuffer = await sharp(buffer)
    .rotate() // auto-orient using EXIF before stripping it
    .resize({
      width: PREVIEW_LONG_EDGE,
      height: PREVIEW_LONG_EDGE,
      fit: "inside",
      withoutEnlargement: true,
    })
    .jpeg({ quality: PREVIEW_QUALITY })
    .toBuffer();
  await fs.writeFile(resolveUploadPath(previewPath), previewBuffer);

  return { storedPath, previewPath, finalMimeType: mimeType };
}

export async function deleteUploadFiles(paths: (string | null)[]) {
  await Promise.all(
    paths
      .filter((p): p is string => Boolean(p))
      .map((p) => fs.unlink(resolveUploadPath(p)).catch(() => undefined)),
  );
}
