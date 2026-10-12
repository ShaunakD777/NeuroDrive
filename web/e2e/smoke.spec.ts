import { expect, test } from "@playwright/test";

test("three panels render and the live socket connects", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByTestId("panel-environment")).toBeVisible();
  await expect(page.getByTestId("panel-network")).toBeVisible();
  await expect(page.getByTestId("panel-experiment")).toBeVisible();

  // The badge shows "connected" only after the server's hello message arrives.
  await expect(page.getByTestId("connection-badge")).toHaveAttribute("data-status", "connected");
});

test("the dividers resize the panels", async ({ page }) => {
  await page.goto("/");

  const environment = page.getByTestId("panel-environment");
  const before = (await environment.boundingBox())!.width;

  const divider = page.getByRole("separator").first();
  await divider.focus();
  await page.keyboard.press("ArrowRight");

  await expect.poll(async () => (await environment.boundingBox())!.width).toBeGreaterThan(before);
});
