import { expect, type Page, test } from "@playwright/test";

// Every test starts from a clean browser: no saved layout, so the defaults apply.

async function chooseLayout(page: Page, name: "Page" | "Split" | "All") {
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name }).click();
}

test("opens on the Environment page and the live socket connects", async ({ page }) => {
  await page.goto("/");

  await expect(page).toHaveURL(/#\/environment$/);
  await expect(page.getByTestId("page-environment")).toBeVisible();
  await expect(page.getByRole("link", { name: "Environment" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  // The badge shows "connected" only after the server's hello message arrives.
  await expect(page.getByTestId("connection-badge")).toHaveAttribute("data-status", "connected");
});

test("tabs and the back button switch pages without reloading or reconnecting", async ({
  page,
}) => {
  const sockets: string[] = [];
  page.on("websocket", (ws) => {
    if (ws.url().endsWith("/ws/live")) sockets.push(ws.url());
  });

  await page.goto("/");
  await expect(page.getByTestId("connection-badge")).toHaveAttribute("data-status", "connected");
  const socketsAfterLoad = sockets.length;
  // A marker on the window object disappears if the document ever reloads.
  await page.evaluate(() => ((window as unknown as { __ndMarker: number }).__ndMarker = 42));

  await page.getByRole("link", { name: "Network" }).click();
  await expect(page).toHaveURL(/#\/network$/);
  await expect(page.getByTestId("page-network")).toBeVisible();

  await page.getByRole("link", { name: "Experiment" }).click();
  await expect(page.getByTestId("page-experiment")).toBeVisible();

  await page.goBack();
  await expect(page).toHaveURL(/#\/network$/);
  await expect(page.getByTestId("page-network")).toBeVisible();

  const marker = await page.evaluate(() => (window as unknown as { __ndMarker?: number }).__ndMarker);
  expect(marker).toBe(42);
  expect(sockets.length).toBe(socketsAfterLoad);
  await expect(page.getByTestId("connection-badge")).toHaveAttribute("data-status", "connected");
});

test("an unknown address falls back to the Environment page", async ({ page }) => {
  await page.goto("/#/garage");
  await expect(page).toHaveURL(/#\/environment$/);
  await expect(page.getByTestId("page-environment")).toBeVisible();
});

test("the companion drawer opens and closes", async ({ page }) => {
  await page.goto("/#/environment");
  const toggle = page.getByTestId("companion-drawer").getByRole("button");

  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByText("Network (compact)")).toBeVisible();

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByText("Network (compact)")).toBeHidden();

  await toggle.click();
  await expect(page.getByText("Network (compact)")).toBeVisible();
});

test("Split shows two views and All shows three; the choice survives a reload", async ({
  page,
}) => {
  await page.goto("/#/environment");

  await chooseLayout(page, "Split");
  await expect(page.getByTestId("split-environment")).toBeVisible();
  await expect(page.getByTestId("split-network")).toBeVisible();
  await page.getByLabel("Second view").selectOption("experiment");
  await expect(page.getByTestId("split-experiment")).toBeVisible();
  await expect(page.getByTestId("split-network")).toHaveCount(0);

  await chooseLayout(page, "All");
  for (const id of ["environment", "network", "experiment"]) {
    await expect(page.getByTestId(`panel-${id}`)).toBeVisible();
  }

  await page.reload();
  await expect(page.getByTestId("panel-network")).toBeVisible();
  await expect(
    page.getByRole("group", { name: "Layout" }).getByRole("button", { name: "All" }),
  ).toHaveAttribute("aria-pressed", "true");
});

test("the dividers resize the panels", async ({ page }) => {
  await page.goto("/#/environment");
  await chooseLayout(page, "All");

  const environment = page.getByTestId("panel-environment");
  const before = (await environment.boundingBox())!.width;

  const divider = page.getByRole("separator").first();
  await divider.focus();
  await page.keyboard.press("ArrowRight");

  await expect.poll(async () => (await environment.boundingBox())!.width).toBeGreaterThan(before);
});
