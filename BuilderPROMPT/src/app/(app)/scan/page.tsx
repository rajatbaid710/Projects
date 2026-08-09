import { AppHeader } from "@/components/nav/app-header";
import { ScanUploader } from "@/components/upload/scan-uploader";

export default function ScanPage() {
  return (
    <div className="flex flex-col">
      <AppHeader title="BillBox" subtitle="Scan an invoice or bill" />
      <ScanUploader />
    </div>
  );
}
