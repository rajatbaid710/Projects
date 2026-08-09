import { AppHeader } from "@/components/nav/app-header";
import { HistoryClient } from "@/components/history/history-client";

export default function HistoryPage() {
  return (
    <div className="flex flex-col">
      <AppHeader title="History" subtitle="Your uploaded documents" />
      <HistoryClient />
    </div>
  );
}
