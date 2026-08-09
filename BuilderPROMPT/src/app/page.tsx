import { redirect } from "next/navigation";

// The proxy guarantees a valid session for every request that reaches this
// page (unauthenticated users are redirected to /login before this renders).
export default function RootPage() {
  redirect("/scan");
}
