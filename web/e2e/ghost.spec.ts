import { expect, type Locator, type Page, test } from "@playwright/test";

// Follow-up to task 2.10: the what-if ghost car (ADR-026) and choosing the car from the Network
// page (drop-down, and clicking a car in the drawer's small track view).

async function randomNetworks(page: Page) {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-policy").selectOption("random_networks");
  await expect(page.getByTestId("select-policy")).toHaveValue("random_networks");
  const box = page.getByTestId("input-cars");
  await box.fill("20");
  await box.press("Enter");
  await expect(page.getByTestId("select-car").locator("option")).toHaveCount(20);
}

/** Car 5 after 12 steps, Steer selected, Environment and Network side by side. */
async function inspectCar5(page: Page) {
  await randomNetworks(page);
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name: "All" }).click();
  await page.getByTestId("select-car").selectOption("4");
  for (let i = 0; i < 12; i++) await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("hud-step")).toHaveText("12");
  await page.getByTestId("net-node-3-0").click();
  await expect(page.getByTestId("whatif-panel")).toBeVisible();
}

const ghostOf = (canvas: Locator) => canvas.getAttribute("data-ghost");

async function carPositions(canvas: Locator): Promise<[number, number][]> {
  const text = (await canvas.getAttribute("data-cars")) ?? "";
  return text === "" ? [] : text.split(",").map((p) => p.split(":").map(Number) as [number, number]);
}

test("moving a what-if slider draws the ghost car; resetting or stepping removes it", async ({
  page,
}) => {
  await inspectCar5(page);
  const canvas = page.getByTestId("environment-canvas").first();
  await expect(canvas).toHaveAttribute("data-ghost", "");
  await expect(page.getByTestId("ghost-legend")).toHaveCount(0);

  await page.getByTestId("whatif-6").fill("0.9"); // heading error far to one side
  await expect.poll(() => ghostOf(canvas)).not.toBe("");
  const [x, y, psi] = ((await ghostOf(canvas)) ?? "").split(",").map(Number);
  expect([x, y, psi].every(Number.isFinite)).toBe(true);
  await expect(page.getByTestId("ghost-legend")).toContainText("What-if ghost");
  await expect(page.getByTestId("ghost-legend")).toContainText("car 5");

  // a different question moves the ghost
  await page.getByTestId("whatif-6").fill("-0.9");
  await expect.poll(() => ghostOf(canvas)).not.toBe(`${x},${y},${psi}`);

  await page.getByTestId("whatif-reset").click();
  await expect(canvas).toHaveAttribute("data-ghost", "");

  await page.getByTestId("whatif-6").fill("0.9");
  await expect.poll(() => ghostOf(canvas)).not.toBe("");
  await page.getByTestId("btn-step").click(); // the moment has passed
  await expect(page.getByTestId("hud-step")).toHaveText("13");
  await expect(canvas).toHaveAttribute("data-ghost", "");
});

test("the ghost also shows in the Network page's drawer while the sliders are moved", async ({
  page,
}) => {
  await randomNetworks(page);
  await page.getByTestId("select-car").selectOption("4");
  for (let i = 0; i < 12; i++) await page.getByTestId("btn-step").click();
  await page.getByRole("link", { name: "Network" }).click();
  await page.getByTestId("net-node-3-0").click();
  await page.getByTestId("whatif-6").fill("0.9");
  const drawerCanvas = page.getByTestId("environment-compact").getByTestId("environment-canvas");
  await expect.poll(() => ghostOf(drawerCanvas)).not.toBe("");
});

test("the Network page's drop-down jumps straight to any car", async ({ page }) => {
  await randomNetworks(page);
  await page.getByRole("link", { name: "Network" }).click();
  const pick = page.getByTestId("car-pick");
  await expect(pick.locator("option")).toHaveCount(20);
  await pick.selectOption("13");
  await expect(page.getByText("The network of car 14")).toBeVisible();
  await expect(page.getByTestId("car-label")).toHaveText("of 20");
  await page.getByTestId("car-next").click();
  await expect(pick).toHaveValue("14");
  await page.getByRole("link", { name: "Environment" }).click();
  await expect(page.getByTestId("select-car")).toHaveValue("14");
});

test("clicking a car in the drawer's track view selects it on the Network page", async ({ page }) => {
  await randomNetworks(page);
  // Seeded random starts spread the 20 cars round the 731 m loop (about 36 m apart), so the
  // drawer's close-up view (it follows the selected car) shows a few of them, well separated.
  await page.getByTestId("check-random-start").click();
  await expect(page.getByTestId("check-random-start")).toBeChecked();
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByRole("link", { name: "Network" }).click();
  const canvas = page.getByTestId("environment-compact").getByTestId("environment-canvas");
  await expect.poll(async () => (await carPositions(canvas)).length).toBe(20);
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  const cars = await carPositions(canvas);
  const inside = ([x, y]: [number, number]) => x > 8 && y > 8 && x < box!.width - 8 && y < box!.height - 8;
  const alone = cars.findIndex(
    (p, i) => i !== 0 && inside(p) && cars.every((q, j) => j === i || Math.hypot(q[0] - p[0], q[1] - p[1]) > 30),
  );
  expect(alone).toBeGreaterThan(0); // at least one other car stands on its own in view
  await canvas.click({ position: { x: cars[alone][0], y: cars[alone][1] } });
  await expect(page.getByTestId("car-pick")).toHaveValue(String(alone));
  await expect(page.getByText(`The network of car ${alone + 1}`)).toBeVisible();
});
