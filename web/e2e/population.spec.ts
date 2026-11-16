import { expect, test, type Locator, type Page } from "@playwright/test";

// Population mode (task 7.6, ADR-057): a whole GA generation races in the Environment view,
// coloured by its fitness rank, with an earlier generation's champion as a ghost; clicking a car
// shows its own network. global-setup.ts trains a 2-generation GA run (tag "e2e").

interface RunSummary {
  run_id: string;
  status: string;
}

async function gaRun(page: Page): Promise<string> {
  const { runs } = (await (await page.request.get("/api/runs")).json()) as { runs: RunSummary[] };
  const run = runs.find((r) => r.run_id.endsWith("-ga-e2e") && r.status === "completed");
  expect(run, "an e2e GA run").toBeTruthy();
  return run!.run_id;
}

async function carPositions(canvas: Locator): Promise<[number, number][]> {
  const text = (await canvas.getAttribute("data-cars")) ?? "";
  return text === "" ? [] : text.split(",").map((p) => p.split(":").map(Number) as [number, number]);
}

test("a GA generation races from the timeline, and any car can be inspected", async ({ page }) => {
  test.setTimeout(90_000); // the race itself takes a few seconds at full speed
  const runId = await gaRun(page);
  await page.goto("/#/experiment");
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name: "All" }).click();
  await page.getByTestId("experiment-tab-timeline").click();
  await expect(page.getByTestId("timeline-run")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("timeline-run").selectOption(runId);
  const slider = page.getByTestId("timeline-slider");
  await expect(slider).toBeVisible();
  await slider.fill(await slider.getAttribute("max") ?? "0"); // the last checkpoint: generation 2
  await expect(page.getByTestId("timeline-label")).toHaveAttribute("data-unit", "2");

  await page.getByTestId("timeline-race").click();
  await expect(page.getByTestId("timeline-race")).toHaveText("Generation 2 races in the Environment view");
  const canvas = page.getByTestId("environment-canvas").first();
  await expect(canvas).toHaveAttribute("data-race", "2:0:1"); // generation 2, start 1, ghost = generation 1
  await expect(page.getByTestId("select-policy").first()).toHaveValue("population");
  await expect(page.getByTestId("select-car").first().locator("option")).toHaveCount(51); // 50 + the ghost
  await expect(page.getByTestId("select-car").first().locator("option").last()).toHaveText("Ghost (generation 1's best)");
  await expect(page.getByTestId("race-legend")).toContainText("Generation 2");
  await expect(page.getByTestId("race-legend")).toContainText("generation 1's champion");
  await expect(page.getByTestId("hud-race")).toHaveAttribute("data-kind", "network");

  // Race for a while at full speed, then pause: the cars have spread out.
  await page.getByTestId("camera-fit").click();
  await page.getByTestId("select-speed").selectOption("8");
  await page.getByTestId("btn-play").click();
  await expect.poll(async () => Number(await page.getByTestId("hud-step").textContent()), { timeout: 20_000 }).toBeGreaterThan(300);
  if ((await page.getByTestId("btn-play").textContent()) === "Pause") await page.getByTestId("btn-play").click();

  // Click a car that sits alone on screen: it becomes the selected car, everywhere.
  const cars = await carPositions(canvas);
  expect(cars).toHaveLength(51);
  const alone = cars.findIndex(
    ([x, y], i) => i > 0 && i < 50 && cars.every(([u, v], j) => j === i || Math.hypot(u - x, v - y) > 12) && x > 5 && y > 5,
  );
  expect(alone, "a car alone on screen").toBeGreaterThan(0);
  const before = await page.getByTestId("net-edge-0-0-0").first().getAttribute("data-weight");
  await canvas.click({ position: { x: cars[alone][0], y: cars[alone][1] } });
  await expect(page.getByTestId("select-car").first()).toHaveValue(String(alone));
  await expect(page.getByTestId("hud-race-role")).toContainText(`car ${alone + 1} · rank `);
  await expect(page.getByText(`The network of car ${alone + 1}`).first()).toBeVisible();
  await expect.poll(() => page.getByTestId("net-edge-0-0-0").first().getAttribute("data-weight")).not.toBe(before);

  // The ghost can be selected too, and shows its generation.
  await page.getByTestId("select-car").first().selectOption("50");
  await expect(page.getByTestId("hud-race")).toHaveAttribute("data-kind", "ghost");
  await expect(page.getByTestId("hud-race-role")).toHaveText("ghost: generation 1's champion");

  // Another start, then no ghost: each reloads the race.
  const start = page.getByTestId("race-start");
  await start.fill("3");
  await start.press("Enter");
  await expect(canvas).toHaveAttribute("data-race", "2:2:1");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  const ghost = page.getByTestId("race-ghost");
  await ghost.fill("0");
  await ghost.press("Enter");
  await expect(canvas).toHaveAttribute("data-race", "2:2:");
  await expect(page.getByTestId("select-car").first().locator("option")).toHaveCount(50);
  await expect(page.getByTestId("race-legend")).not.toContainText("ghost");

  // Generation 1 has no earlier generation, so no ghost can be chosen.
  const generation = page.getByTestId("race-generation");
  await generation.fill("1");
  await generation.press("Enter");
  await expect(canvas).toHaveAttribute("data-race", "1:2:");
  await expect(page.getByTestId("race-ghost")).toBeDisabled();
});
