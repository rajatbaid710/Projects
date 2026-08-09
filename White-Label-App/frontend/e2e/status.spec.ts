import { expect, test } from "@playwright/test";

/**
 * Phase 0 E2E: the status page is currently the only user-facing flow.
 *
 * Thin on purpose — it exists so the Playwright harness itself is proven working
 * before there is anything substantial to test. It gets replaced by real flows
 * (sign up → verify → log in, upload → chat) as the phases land.
 *
 * Requires the full stack: data services, API, and web client.
 */

test.describe("status page", () => {
  test("renders the product name served by the API", async ({ page }) => {
    await page.goto("/");

    // Branding comes from GET /api/v1/meta, never from a constant in the web
    // app — this assertion is what would catch a regression to a hardcoded name.
    const heading = page.getByRole("heading", { level: 1 });
    await expect(heading).toBeVisible();
    await expect(heading).not.toBeEmpty();
  });

  test("reports every backing service as healthy", async ({ page }) => {
    await page.goto("/");

    for (const service of ["postgres", "redis", "qdrant"]) {
      const row = page.getByRole("listitem").filter({ hasText: service });
      await expect(row).toBeVisible();
      // "unreachable" is what the page renders for a failed dependency; a
      // latency figure means the probe succeeded.
      await expect(row).not.toContainText("unreachable");
    }
  });

  test("refresh re-queries the API", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("listitem").first()).toBeVisible();

    const readiness = page.waitForResponse(
      (response) => response.url().includes("/health/ready") && response.status() < 500,
    );
    await page.getByRole("button", { name: "Refresh" }).click();
    await readiness;

    await expect(page.getByRole("listitem")).toHaveCount(3);
  });

  test("shows an actionable error when the API is unreachable", async ({ page }) => {
    // The failure path matters as much as the happy one: a developer whose
    // backend is down should be told what to start, not shown a blank page.
    await page.route("**/api/v1/meta", (route) => route.abort("failed"));
    await page.route("**/health/ready", (route) => route.abort("failed"));

    await page.goto("/");

    await expect(page.getByText(/cannot reach the api/i)).toBeVisible();
    await expect(page.getByText(/docker compose/i)).toBeVisible();
  });
});
