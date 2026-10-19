import { expect, type Page, test } from "@playwright/test";

// The output inspector, what-if and ray colours (task 2.10). Random networks, 20 cars, all on the
// start line (no seed: networks from seed 0). Car 5 is a seed-0 brain that drives off and reacts
// to its rays, so its sensitivities are not all tiny.

async function inspectCar5Steer(page: Page) {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-policy").selectOption("random_networks");
  await expect(page.getByTestId("select-policy")).toHaveValue("random_networks");
  const box = page.getByTestId("input-cars");
  await box.fill("20");
  await box.press("Enter");
  await expect(page.getByTestId("select-car").locator("option")).toHaveCount(20);
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name: "All" }).click();
  await page.getByTestId("select-car").selectOption("4");
  await expect(page.getByText("The network of car 5")).toBeVisible();
  for (let i = 0; i < 12; i++) await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("hud-step")).toHaveText("12");
  await expect(page.getByTestId("inspection-hint")).toBeVisible();
  await page.getByTestId("net-node-3-0").click(); // Steer
  await expect(page.getByTestId("output-inspector")).toBeVisible();
}

const num = async (page: Page, testId: string, attribute = "data-value") =>
  Number(await page.getByTestId(testId).first().getAttribute(attribute));

test("clicking Steer opens the inspector: value in words, receipt, sensitivity", async ({ page }) => {
  await inspectCar5Steer(page);
  await expect(page.getByTestId("inspector-headline")).toHaveText(/^Steer = −?\d\.\d\d → /);
  await expect(page.locator('[data-testid^="receipt-row-"]')).toHaveCount(6);
  await expect(page.locator('[data-testid^="sensitivity-"]')).toHaveCount(7);
  await expect(page.getByTestId("inspector-sum")).toContainText("tanh(z) = ");
  await expect(page.getByTestId("inspector-nudge")).toContainText("rose by 0.1, steer would change by about");
  // the inspector's value is the diagram's steer output
  const steer = await num(page, "net-output-0");
  await expect(page.getByTestId("inspector-sum")).toContainText(Math.abs(steer).toFixed(3));

  await page.getByTestId("inspect-throttle").click();
  await expect(page.getByTestId("inspector-headline")).toHaveText(/^Throttle = /);
  await expect(page.getByTestId("net-node-3-1")).toHaveAttribute("aria-pressed", "true");
});

test("paused, dragging the left-ray slider changes steer as the sensitivity predicts", async ({
  page,
}) => {
  await inspectCar5Steer(page);
  const s = await num(page, "sensitivity-0"); // steer's sensitivity to the far-left ray
  expect(Math.abs(s)).toBeGreaterThan(0.01);
  const ray = await num(page, "net-input-0");
  await page.getByTestId("whatif-0").fill((ray + 0.1).toFixed(2));
  const row = page.getByTestId("whatif-out-0");
  await expect(row).toBeVisible();
  const now = Number(await row.getAttribute("data-now"));
  const tried = Number(await row.getAttribute("data-tried"));
  const predicted = Number(await row.getAttribute("data-predicted"));
  const moved = (ray + 0.1).toFixed(2) === "1.00" ? 1 - ray : Number((ray + 0.1).toFixed(2)) - ray;
  // sign: the network moved the way the sensitivity said
  expect(Math.sign(tried - now)).toBe(Math.sign(s));
  // rough size: the prediction is the straight line s * change, and the real answer is close
  expect(predicted - now).toBeCloseTo(s * moved, 2);
  expect(Math.abs(tried - predicted)).toBeLessThan(0.5 * Math.abs(predicted - now));
  // the slider shows its changed value; "Back to the real inputs" clears it
  await expect(page.getByTestId("whatif-value-0")).toHaveClass(/whatif-changed/);
  await page.getByTestId("whatif-reset").click();
  await expect(page.getByTestId("whatif-result")).toHaveCount(0);
  await expect(page.getByTestId("whatif-value-0")).not.toHaveClass(/whatif-changed/);
});

test("the rays are coloured by the selected output's sensitivity", async ({ page }) => {
  await inspectCar5Steer(page);
  const canvas = page.getByTestId("environment-canvas").first();
  await expect(page.getByTestId("ray-legend")).toContainText("how steer reacts");
  const attribute = (await canvas.getAttribute("data-ray-sensitivity")) ?? "";
  const rays = attribute.split(",").map(Number);
  expect(rays).toHaveLength(5);
  for (let k = 0; k < 5; k++) expect(rays[k]).toBeCloseTo(await num(page, `sensitivity-${k}`), 4);

  await page.getByTestId("net-node-3-0").click(); // unselect steer
  await expect(canvas).toHaveAttribute("data-ray-sensitivity", "");
  await expect(page.getByTestId("ray-legend")).toHaveCount(0);
});

test("what-if is only for a paused car, and a new step forgets it", async ({ page }) => {
  await inspectCar5Steer(page);
  const ray = await num(page, "net-input-0");
  await page.getByTestId("whatif-0").fill(Math.min(1, ray + 0.2).toFixed(2));
  await expect(page.getByTestId("whatif-result")).toBeVisible();

  await page.getByTestId("btn-step").click(); // the car moves on: the question no longer applies
  await expect(page.getByTestId("hud-step")).toHaveText("13");
  await expect(page.getByTestId("whatif-result")).toHaveCount(0);
  await expect(page.getByTestId("whatif-value-0")).not.toHaveClass(/whatif-changed/);

  await page.getByTestId("btn-play").click();
  await expect(page.getByTestId("whatif-paused-hint")).toBeVisible();
  await expect(page.locator('[data-testid^="whatif-"][type="range"]')).toHaveCount(0);
});
