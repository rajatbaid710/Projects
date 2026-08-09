"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ThemeToggle } from "@/components/providers/theme-toggle";
import { formatCurrency } from "@/components/review/field-utils";

type Usage = {
  totalCostUsd: number;
  monthCostUsd: number;
  totalDocuments: number;
  model: string;
};

export function SettingsClient({ email }: { email: string }) {
  const router = useRouter();
  const [usage, setUsage] = useState<Usage | null>(null);
  const [loggingOut, setLoggingOut] = useState(false);

  useEffect(() => {
    fetch("/api/usage/summary")
      .then((res) => res.json())
      .then(setUsage)
      .catch(() => undefined);
  }, []);

  async function handleLogout() {
    setLoggingOut(true);
    try {
      const res = await fetch("/api/auth/logout", { method: "POST" });
      if (!res.ok) throw new Error("logout failed");
      router.push("/login");
      router.refresh();
    } catch {
      toast.error("Could not sign out. Please try again.");
      setLoggingOut(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 px-4 pb-8 pt-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Account</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Signed in as</p>
          <p className="font-medium">{email}</p>
          <Button
            variant="outline"
            className="mt-4 w-full gap-2"
            onClick={handleLogout}
            disabled={loggingOut}
          >
            <LogOut className="h-4 w-4" />
            {loggingOut ? "Signing out…" : "Sign out"}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Appearance</CardTitle>
        </CardHeader>
        <CardContent>
          <ThemeToggle />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">AI usage</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {usage ? (
            <>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Extraction model</span>
                <span className="font-mono text-xs">{usage.model}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">This month</span>
                <span className="font-medium">
                  {formatCurrency(usage.monthCostUsd, "USD")}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">All time ({usage.totalDocuments} docs)</span>
                <span className="font-medium">{formatCurrency(usage.totalCostUsd, "USD")}</span>
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Loading…</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
