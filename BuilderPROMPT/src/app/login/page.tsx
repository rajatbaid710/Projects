"use client";

import { useActionState, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ReceiptText } from "lucide-react";
import { loginAction, type LoginActionState } from "@/lib/auth/actions";

const initialState: LoginActionState = { error: null };

export default function LoginPage() {
  const [state, formAction, pending] = useActionState(loginAction, initialState);
  const [next, setNext] = useState("/scan");

  // Read once on mount — a hidden field mirrors this into the form so the
  // Server Action gets it even on a pre-hydration native form submission.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const value = params.get("next");
    // window.location doesn't exist during SSR, so this can only run
    // client-side after mount — not derivable during render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (value && value.startsWith("/")) setNext(value);
  }, []);

  return (
    <div className="flex min-h-dvh items-center justify-center bg-muted/30 px-4">
      <Card className="w-full max-w-sm border-none shadow-lg sm:border sm:shadow-sm">
        <CardHeader className="items-center text-center">
          <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
            <ReceiptText className="h-6 w-6" />
          </div>
          <CardTitle className="text-xl">BillBox</CardTitle>
          <CardDescription>
            Enter your email to track your scanned invoices and bills.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form action={formAction} className="space-y-4">
            <input type="hidden" name="next" value={next} />
            <div className="space-y-2">
              <Label htmlFor="email">Email address</Label>
              <Input
                id="email"
                name="email"
                type="email"
                inputMode="email"
                autoComplete="email"
                autoFocus
                required
                placeholder="you@business.com"
                className="h-12 text-base"
              />
            </div>
            {state.error && <p className="text-sm text-destructive">{state.error}</p>}
            <Button type="submit" className="h-12 w-full text-base" disabled={pending}>
              {pending ? "Continuing…" : "Continue"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
