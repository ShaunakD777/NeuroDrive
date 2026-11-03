import { expect, test, type Page } from "@playwright/test";

// The Training tab (task 5.8, ADR-041): a run launched from the app trains as a separate process
// (`nd train`), and its chart grows while it trains, not only at the end. Each run of these tests
// adds two small random-search run folders (tags "e2elaunch" and "e2estop") to runs/.

async function openTraining(page: Page) {
  await page.goto("/#/experiment");
  await page.getByTestId("experiment-tab-training").click();
  await expect(page.getByTestId("training")).toBeVisible();
  // The config list and the /ws/runs hello arrive within a second, but the first page of a test
  // worker can wait several seconds for the dev server to compile changed modules.
  await expect(page.getByTestId("launch-start")).toBeEnabled({ timeout: 15_000 });
}

async function launch(page: Page, tag: string, units: number): Promise<string> {
  await page.getByTestId("launch-config").selectOption("random_search.yaml");
  await page.getByTestId("launch-tag").fill(tag);
  await page.getByTestId("launch-length").fill(String(units));
  await page.getByTestId("launch-start").click();
  const watched = page.getByTestId("watched-run").locator("strong");
  await expect(watched).toHaveText(new RegExp(`-random_search-${tag}$`), { timeout: 30_000 });
  return (await watched.textContent())!;
}

test("a run launched from the app trains in the background and its chart grows unit by unit", async ({ page }) => {
  test.setTimeout(180_000);
  await openTraining(page);
  const runId = await launch(page, "e2elaunch", 3);
  const row = page.locator(`[data-testid="training-run"][data-run-id="${runId}"]`);
  await expect(row).toBeVisible();

  // Record every point count the chart shows until the run has finished.
  const chart = page.getByTestId("live-chart");
  const seen = new Set<number>();
  const deadline = Date.now() + 150_000;
  while ((await row.getAttribute("data-status")) !== "completed") {
    expect(Date.now(), "the run finishes").toBeLessThan(deadline);
    if (await chart.isVisible()) seen.add(Number(await chart.getAttribute("data-points")));
    await page.waitForTimeout(100);
  }
  await expect(chart).toHaveAttribute("data-points", "3");
  seen.add(3);
  // One point per unit, each seen on its own: the chart grew while the run trained.
  expect([...seen].sort()).toEqual([1, 2, 3]);
  await expect(page.getByTestId("live-point")).toHaveCount(3);
  await expect(chart).toHaveAttribute("data-metric", "best_fitness");
  await chart.locator("rect.toy-hit").hover(); // an HTML read-out, sharp at any chart size
  await expect(page.getByTestId("live-hover")).toContainText(/^unit \dbest fitness \d/);
  expect(await page.getByTestId("live-hover").evaluate((el) => el.tagName)).toBe("DIV");

  await expect(row.getByTestId("training-run-units")).toHaveText("3 / 3 units");
  await expect(row.getByTestId("training-run-status")).toHaveText("completed");
  await expect(row.getByTestId("training-stop")).toHaveCount(0);
  // Every checkpoint (0, 1, 2 and the last unit, 3) was examined; its row appeared in the table.
  await expect(page.getByTestId("exam-row")).toHaveCount(4);
  await expect(page.getByTestId("exam-row").last()).toHaveAttribute("data-unit", "3");
  await expect(page.getByTestId("live-exam-point")).toHaveCount(3); // units 1-3 (0 has no line)

  // The folder is what the page showed: 3 metrics lines, status completed.
  const detail = await (await page.request.get(`/api/runs/${runId}`)).json();
  expect(detail.meta.status).toBe("completed");
  expect(detail.meta.stop_reason).toBe("units");
});

test("Stop ends a launched run after its current unit, with its last unit examined", async ({ page }) => {
  test.setTimeout(180_000);
  await openTraining(page);
  const runId = await launch(page, "e2estop", 200);
  const row = page.locator(`[data-testid="training-run"][data-run-id="${runId}"]`);
  await expect(page.getByTestId("live-chart")).toBeVisible({ timeout: 60_000 }); // the first unit
  await row.getByTestId("training-stop").click();
  await expect(row.getByTestId("training-run-status")).toHaveText("stopped", { timeout: 90_000 });
  const detail = await (await page.request.get(`/api/runs/${runId}`)).json();
  expect(detail.meta.stop_reason).toBe("stopped");
  const units = Number(await page.getByTestId("live-chart").getAttribute("data-points"));
  expect(units).toBeGreaterThanOrEqual(1);
  expect(units).toBeLessThan(200);
  // The last unit got a checkpoint and an exam card, like the end of any run.
  const { units: saved } = await (await page.request.get(`/api/runs/${runId}/checkpoints`)).json();
  expect(saved[saved.length - 1]).toBe(units);
  await expect(page.getByTestId("exam-row").last()).toHaveAttribute("data-unit", String(units));
});
