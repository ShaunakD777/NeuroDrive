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

// ── Environment view (task 1.8) ─────────────────────────────────────────────────────────────

async function numberOf(page: Page, testId: string, attribute = "data-value"): Promise<number> {
  return Number(await page.getByTestId(testId).getAttribute(attribute));
}

async function stepOf(page: Page): Promise<number> {
  return Number(await page.getByTestId("hud-step").innerText());
}

test("the Environment page loads the track and draws the car and the HUD", async ({ page }) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await expect(page.getByTestId("select-track")).toHaveValue("train_loop_01");
  await expect(page.getByTestId("hud-status")).toHaveText("Running");
  expect(await numberOf(page, "hud-speed")).toBe(0);
  for (let k = 0; k < 5; k++) await expect(page.getByTestId(`ray-${k}`)).toBeVisible();

  // The canvas is not blank: it holds road, edges and a car, so several different colours.
  await expect
    .poll(async () =>
      page.getByTestId("environment-canvas").evaluate((canvas: HTMLCanvasElement) => {
        const data = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height).data;
        const colours = new Set<number>();
        for (let i = 0; i < data.length; i += 4 * 13) {
          colours.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
        }
        return colours.size;
      }),
    )
    .toBeGreaterThan(4);
});

test("Play, Pause and Step control the simulation", async ({ page }) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  const play = page.getByTestId("btn-play");

  await play.click();
  await expect(play).toHaveText("Pause");
  await expect.poll(() => stepOf(page)).toBeGreaterThan(5);

  await play.click();
  await expect(play).toHaveText("Play");
  await expect(page.getByTestId("btn-step")).toBeEnabled();
  const paused = await stepOf(page);
  await page.waitForTimeout(400);
  expect(await stepOf(page)).toBe(paused);

  await page.getByTestId("btn-step").click();
  await expect.poll(() => stepOf(page)).toBe(paused + 1);
});

test("holding the up arrow accelerates; the steering keys move the steer bar", async ({
  page,
}) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("btn-play").click();

  await page.keyboard.down("ArrowUp");
  await expect.poll(() => numberOf(page, "hud-speed")).toBeGreaterThan(4);
  await expect.poll(async () => Number(await page.getByTestId("hud-throttle").innerText())).toBe(1);

  await page.keyboard.down("ArrowRight");
  await expect.poll(async () => Number(await page.getByTestId("hud-steer").innerText())).toBe(1);
  await page.keyboard.up("ArrowRight");
  await page.keyboard.down("ArrowLeft");
  await expect.poll(async () => Number(await page.getByTestId("hud-steer").innerText())).toBe(-1);
  await page.keyboard.up("ArrowLeft");
  await page.keyboard.up("ArrowUp");
  await expect.poll(async () => Number(await page.getByTestId("hud-throttle").innerText())).toBe(0);
});

test("driving straight off the road ends the episode with a reason; Reset starts over", async ({
  page,
}) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-speed").selectOption("8");
  await page.getByTestId("btn-play").click();
  await page.keyboard.down("ArrowUp");

  // Rays shorten as the walls get close: the nearest ray ends up well under a car's width of road.
  const banner = page.getByTestId("status-banner");
  await expect(banner).toContainText("Crashed", { timeout: 30_000 });
  await page.keyboard.up("ArrowUp");
  await expect(page.getByTestId("hud-status")).toHaveText("Crashed");
  const rays = await Promise.all([0, 1, 2, 3, 4].map((k) => numberOf(page, `ray-${k}`)));
  expect(Math.min(...rays)).toBeLessThan(6);
  await expect(page.getByTestId("btn-play")).toBeDisabled();

  await page.getByTestId("btn-reset").click();
  await expect(banner).toHaveCount(0);
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await expect(page.getByTestId("hud-status")).toHaveText("Running");
  await expect(page.getByTestId("btn-play")).toBeEnabled();
});

test("the compact Environment view shows in the Network page's drawer", async ({ page }) => {
  await page.goto("/#/network");
  const compact = page.getByTestId("environment-compact");
  await expect(compact).toBeVisible();
  await expect(compact.getByTestId("hud-status")).toHaveText("Running");
  await expect(compact.getByTestId("environment-canvas")).toBeVisible();
});

test("moving between pages keeps the same live session", async ({ page }) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("btn-step").click();
  await page.getByTestId("btn-step").click();
  await expect.poll(() => stepOf(page)).toBe(2);

  await page.getByRole("link", { name: "Experiment" }).click();
  await page.getByRole("link", { name: "Environment" }).click();
  await expect(page.getByTestId("hud-step")).toHaveText("2");
});
