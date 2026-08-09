import { AppHeader } from "@/components/nav/app-header";
import { getSession } from "@/lib/auth/session";
import { SettingsClient } from "./settings-client";

export default async function SettingsPage() {
  const session = await getSession();

  return (
    <div className="flex flex-col">
      <AppHeader title="Settings" />
      <SettingsClient email={session?.email ?? ""} />
    </div>
  );
}
