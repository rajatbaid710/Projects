"use server";

import { redirect } from "next/navigation";
import { findOrCreateUser, isValidEmail, normalizeEmail } from "./user";
import { setSessionCookie } from "./session";

export type LoginActionState = { error: string | null };

/**
 * Server Action backing the login form. Deliberately not a client-side
 * fetch() + router.push(): a <form action={...}> submits correctly even if
 * the page hasn't finished hydrating yet (e.g. a fast tap right after the
 * page appears, or slow JS parsing on a phone over a real network) — a
 * fetch-based handler simply doesn't run in that case, and the click falls
 * through to a native (broken) form submission instead.
 */
export async function loginAction(
  _prevState: LoginActionState,
  formData: FormData,
): Promise<LoginActionState> {
  const rawEmail = formData.get("email");
  const rawNext = formData.get("next");
  const email = typeof rawEmail === "string" ? normalizeEmail(rawEmail) : "";

  if (!email || !isValidEmail(email)) {
    return { error: "Enter a valid email address." };
  }

  const user = await findOrCreateUser(email);
  await setSessionCookie({ sub: user.id, email: user.email });

  const next = typeof rawNext === "string" && rawNext.startsWith("/") ? rawNext : "/scan";
  redirect(next);
}
