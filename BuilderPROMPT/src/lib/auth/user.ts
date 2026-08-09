import "server-only";

import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { users } from "@/lib/db/schema";

export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

/** Finds the user by email, creating one on first sign-in. Tracks lastLoginAt. */
export async function findOrCreateUser(rawEmail: string): Promise<{ id: string; email: string }> {
  const email = normalizeEmail(rawEmail);

  const [existing] = await db.select().from(users).where(eq(users.email, email)).limit(1);
  if (existing) {
    await db.update(users).set({ lastLoginAt: new Date() }).where(eq(users.id, existing.id));
    return { id: existing.id, email };
  }

  const [created] = await db
    .insert(users)
    .values({ email, lastLoginAt: new Date() })
    .returning({ id: users.id });

  return { id: created.id, email };
}
