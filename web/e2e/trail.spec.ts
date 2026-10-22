import { expect, type Locator, type Page, test } from "@playwright/test";

// Task 2.11: the decision trail behind the selected car, coloured by the selected output, and
// looking back at an earlier step (click a trail point or drag "Look back"): the Network view then
// shows the trace at that step, recomputed by the Python server from what it remembered.

async function randomNetworksCar5(page: Page) {
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
}

const value = async (page: Page, testId: string) =>
  Number(await page.getByTestId(testId).first().getAttribute("data-value"));

/** Every input and output value the diagram shows, and Steer's sensitivities (when open). */
async function snapshot(page: Page, withSensitivity: boolean): Promise<number[]> {
  const ids = [
    ...Array.from({ length: 7 }, (_, k) => `net-input-${k}`),
    "net-output-0",
    "net-output-1",
    ...(withSensitivity ? Array.from({ length: 7 }, (_, k) => `sensitivity-${k}`) : []),
  ];
  return Promise.all(ids.map((id) => value(page, id)));
}

/** Step once and remember what the diagram shows at the new step. */
async function stepAndRecord(page: Page, step: number, withSensitivity = false) {
  await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("hud-step")).toHaveText(String(step));
  await expect(page.getByTestId("trail-readout")).toContainText(`step ${step}`);
  return snapshot(page, withSensitivity);
}

async function trailPoints(canvas: Locator): Promise<{ step: number; x: number; y: number }[]> {
  const text = (await canvas.getAttribute("data-trail")) ?? "";
  if (text === "") return [];
  return text.split(",").map((p) => {
    const [step, x, y] = p.split(":").map(Number);
    return { step, x, y };
  });
}

test("the trail grows one point per step and is coloured by the selected output", async ({ page }) => {
  await randomNetworksCar5(page);
  const canvas = page.getByTestId("environment-canvas").first();
  const steer: number[] = [];
  const throttle: number[] = [];
  for (let t = 1; t <= 10; t++) {
    // the network's outputs at step t - 1 are the action it took there: trail point t - 1
    steer.push(await value(page, "net-output-0"));
    throttle.push(await value(page, "net-output-1"));
    await page.getByTestId("btn-step").click();
    await expect(page.getByTestId("hud-step")).toHaveText(String(t));
  }
  await expect.poll(async () => (await trailPoints(canvas)).map((p) => p.step)).toEqual([...Array(10).keys()]);
  await expect(page.getByTestId("trail-legend")).toContainText("Click Steer or Throttle");
  await expect(canvas).toHaveAttribute("data-trail-values", "");

  await page.getByTestId("net-node-3-0").click(); // Steer
  await expect(page.getByTestId("trail-legend")).toContainText("Trail coloured by steer");
  await expect.poll(async () => ((await canvas.getAttribute("data-trail-values")) ?? "").split(",").length).toBe(10);
  const coloured = ((await canvas.getAttribute("data-trail-values")) ?? "").split(",").map(Number);
  coloured.forEach((v, t) => expect(v).toBeCloseTo(steer[t], 2));

  await page.getByTestId("inspect-throttle").click();
  await expect(page.getByTestId("trail-legend")).toContainText("Trail coloured by throttle");
  await expect
    .poll(async () => ((await canvas.getAttribute("data-trail-values")) ?? "").split(",").map(Number)[3])
    .toBeCloseTo(throttle[3], 2);
});

test("looking back at a step shows exactly the trace the network had then; Back to now restores it", async ({
  page,
}) => {
  await randomNetworksCar5(page);
  await page.getByTestId("net-node-3-0").click(); // Steer, so the sensitivities show too
  await expect(page.getByTestId("output-inspector")).toBeVisible();
  const seen: number[][] = [await snapshot(page, true)];
  for (let t = 1; t <= 8; t++) seen.push(await stepAndRecord(page, t, true));

  const canvas = page.getByTestId("environment-canvas").first();
  await page.getByTestId("trail-slider").fill("3");
  await expect(page.getByTestId("past-banner")).toContainText("Showing step 3 of car 5");
  await expect(canvas).toHaveAttribute("data-moment", "3");
  await expect(page.getByTestId("trail-readout")).toContainText("Step 3 (0.2 s ago)");
  await expect.poll(() => snapshot(page, true)).toEqual(seen[3]);
  await expect(page.getByTestId("hud-step")).toHaveText("3"); // the HUD shows step 3 too
  await expect(page.getByTestId("hud-past")).toHaveText("Looking back: now is step 8"); // not moved

  await page.getByTestId("trail-slider").fill("0");
  await expect.poll(() => snapshot(page, true)).toEqual(seen[0]);

  await page.getByTestId("trail-live").click();
  await expect(page.getByTestId("past-banner")).toHaveCount(0);
  await expect(canvas).toHaveAttribute("data-moment", "");
  await expect.poll(() => snapshot(page, true)).toEqual(seen[8]);
});

test("clicking a point on the trail jumps the views to that step", async ({ page }) => {
  await randomNetworksCar5(page);
  await page.getByTestId("select-speed").selectOption("4");
  await page.getByTestId("btn-play").click();
  await page.waitForTimeout(1500); // about 180 steps
  await page.getByTestId("btn-play").click(); // Pause
  await expect(page.getByTestId("btn-play")).toHaveText("Play");
  await expect(page.getByTestId("trail-readout")).toContainText("Showing now"); // layout settled
  // Random brains are slow (car 5 goes about 9 m): zoom in so its trail spreads out on screen.
  for (let i = 0; i < 5; i++) await page.getByTestId("zoom-in").click();
  const canvas = page.getByTestId("environment-canvas").first();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  const cars = ((await canvas.getAttribute("data-cars")) ?? "").split(",").map((p) => p.split(":").map(Number));
  const points = await trailPoints(canvas);
  // a trail point well inside the picture and well away from every car (a click on a car picks it)
  const target = points.find(
    (p) =>
      p.x > 20 &&
      p.y > 20 &&
      p.x < box!.width - 20 &&
      p.y < box!.height - 20 &&
      cars.every(([cx, cy]) => Math.hypot(p.x - cx, p.y - cy) > 30),
  );
  expect(target).toBeDefined();
  await canvas.click({ position: { x: target!.x, y: target!.y } });
  // the click picks the nearest point, which may be a neighbour of the one aimed at
  await expect(canvas).not.toHaveAttribute("data-moment", "");
  const step = Number(await canvas.getAttribute("data-moment"));
  expect(Math.abs(step - target!.step)).toBeLessThanOrEqual(3);
  await expect(page.getByTestId("past-banner")).toContainText(`Showing step ${step} of car 5`);
  await expect(page.getByTestId("trail-slider")).toHaveValue(String(step));
});

test("looking back needs a pause and ends with the next step; what-if waits for now", async ({ page }) => {
  await randomNetworksCar5(page);
  for (let t = 1; t <= 4; t++) await stepAndRecord(page, t);
  await page.getByTestId("net-node-3-0").click();
  await page.getByTestId("trail-slider").fill("1");
  await expect(page.getByTestId("past-banner")).toBeVisible();
  await expect(page.getByTestId("whatif-past-hint")).toBeVisible();

  await page.getByTestId("btn-step").click(); // the simulation moves on: back to now
  await expect(page.getByTestId("hud-step")).toHaveText("5");
  await expect(page.getByTestId("past-banner")).toHaveCount(0);
  await expect(page.getByTestId("whatif-6")).toBeVisible();

  await page.getByTestId("btn-play").click();
  await expect(page.getByTestId("trail-slider")).toBeDisabled();
  await expect(page.getByTestId("trail-readout")).toContainText("Pause to look back");
  await page.getByTestId("btn-play").click(); // Pause
  await expect(page.getByTestId("btn-play")).toHaveText("Play");
  await expect(page.getByTestId("trail-slider")).toBeEnabled();
});

test("while looking back the HUD shows that step: speed, status, action, progress and rays", async ({ page }) => {
  await randomNetworksCar5(page);
  const hud = async () => ({
    step: await page.getByTestId("hud-step").textContent(),
    speed: await page.getByTestId("hud-speed").getAttribute("data-value"),
    status: await page.getByTestId("hud-status").textContent(),
    progress: await page.getByTestId("hud-progress").textContent(),
    rays: await Promise.all([0, 1, 2, 3, 4].map((k) => page.getByTestId(`ray-${k}`).getAttribute("data-value"))),
    steer: await page.getByTestId("hud-steer").textContent(),
    throttle: await page.getByTestId("hud-throttle").textContent(),
  });
  const seen = [await hud()];
  for (let t = 1; t <= 6; t++) {
    await page.getByTestId("btn-step").click();
    await expect(page.getByTestId("hud-step")).toHaveText(String(t));
    seen.push(await hud());
  }
  await page.getByTestId("trail-slider").fill("2");
  await expect(page.getByTestId("hud-past")).toHaveText("Looking back: now is step 6");
  const past = await hud();
  // everything describes step 2; the action is the one chosen there, which the live HUD showed
  // one step later (live, the HUD shows the last action applied)
  expect({ ...past, steer: null, throttle: null }).toEqual({ ...seen[2], steer: null, throttle: null });
  expect([past.steer, past.throttle]).toEqual([seen[3].steer, seen[3].throttle]);

  await page.getByTestId("trail-live").click();
  await expect(page.getByTestId("hud-past")).toHaveCount(0);
  expect(await hud()).toEqual(seen[6]);
});
