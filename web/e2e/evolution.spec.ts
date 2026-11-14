import { expect, test, type Page } from "@playwright/test";

// The evolution charts of a GA run (task 7.7, ADR-058): fitness lines, the fitness distribution as
// a heat map with the chosen generation's histogram, and the diversity, in the Timeline tab.
// global-setup.ts trains a 2-generation GA run of 50 networks (tag "e2e").

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

test("a GA run's evolution charts show every generation, and a click chooses one", async ({ page }) => {
  const runId = await gaRun(page);
  const answer = await (await page.request.get(`/api/runs/${runId}/evolution`)).json();
  expect(answer.unit).toEqual([1, 2]);
  expect(answer.counts.map((row: number[]) => row.reduce((a, b) => a + b, 0))).toEqual([50, 50]);

  await page.goto("/#/experiment");
  await page.getByTestId("experiment-tab-timeline").click();
  await expect(page.getByTestId("timeline-run")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("timeline-run").selectOption(runId);

  const panel = page.getByTestId("evolution");
  await expect(panel).toHaveAttribute("data-generations", "2");
  await expect(page.getByTestId("evolution-fitness")).toHaveAttribute("data-points", "2");
  await expect(page.getByTestId("evolution-diversity")).toHaveAttribute("data-points", "2");
  for (const key of ["best", "median", "mean", "worst"]) await expect(page.getByTestId(`evolution-line-${key}`)).toHaveCount(1);
  await expect(page.getByTestId("evolution-fitness-legend")).toContainText("median");
  // every network of both generations sits in one cell of the heat map
  await expect(page.getByTestId("evolution-distribution")).toHaveAttribute("data-networks", "100");
  const cells = page.getByTestId("evolution-cell");
  const counts = await cells.evaluateAll((els) => els.map((e) => Number(e.getAttribute("data-count"))));
  expect(counts.reduce((a, b) => a + b, 0)).toBe(100);

  // The panel follows the slider: the last checkpoint is generation 2.
  const slider = page.getByTestId("timeline-slider");
  await slider.fill((await slider.getAttribute("max")) ?? "0");
  await expect(panel).toHaveAttribute("data-generation", "2");
  await expect(page.getByTestId("evolution-histogram")).toHaveAttribute("data-generation", "2");
  const bars = await page.getByTestId("evolution-bar").evaluateAll((els) => els.map((e) => Number(e.getAttribute("data-count"))));
  expect(bars.reduce((a, b) => a + b, 0)).toBe(50);

  // Hover shows the numbers of the generation under the pointer.
  const hit = page.getByTestId("evolution-fitness-hit");
  const box = (await hit.boundingBox())!;
  await hit.hover({ position: { x: box.width * 0.1, y: box.height / 2 } });
  await expect(page.getByTestId("evolution-fitness-hover")).toContainText("generation 1");
  await expect(page.getByTestId("evolution-fitness-hover")).toContainText("best so far");

  // Clicking generation 1 in the heat map chooses it: the histogram shows it, and it can race.
  const map = page.getByTestId("evolution-distribution-hit");
  const mapBox = (await map.boundingBox())!;
  const bottomLeft = { x: mapBox.width * 0.1, y: mapBox.height - 3 };
  await map.hover({ position: bottomLeft });
  await expect(page.getByTestId("evolution-distribution-hover")).toContainText("network");
  await map.click({ position: bottomLeft });
  await expect(panel).toHaveAttribute("data-generation", "1");
  await expect(page.getByTestId("evolution-histogram")).toHaveAttribute("data-generation", "1");
  await page.getByTestId("evolution-race").click();
  await expect(page.getByTestId("evolution-race")).toHaveText("Generation 1 races from start 1 in the Environment view");
});

// One start at a time (task 7.7 follow-up, ADR-059, api v1.24): the start picker, the starts map
// and the outcome grid, whose cells race the generation from that start with that car selected.
test("one start at a time: the picker, the starts map and the outcome grid race the right car", async ({ page }) => {
  test.setTimeout(60_000);
  const runId = await gaRun(page);
  const every = await (await page.request.get(`/api/runs/${runId}/evolution`)).json();
  expect(every.n_starts).toBe(10);
  const onStart2 = await (await page.request.get(`/api/runs/${runId}/evolution?start=1`)).json();
  expect(onStart2.start).toBe(1);
  const grid = await (await page.request.get(`/api/runs/${runId}/generations/2/starts`)).json();
  expect(grid.order).toHaveLength(50);

  await page.goto("/#/experiment");
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name: "All" }).click();
  await page.getByTestId("experiment-tab-timeline").click();
  await expect(page.getByTestId("timeline-run")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("timeline-run").selectOption(runId);
  const slider = page.getByTestId("timeline-slider");
  await slider.fill((await slider.getAttribute("max")) ?? "0"); // generation 2
  const panel = page.getByTestId("evolution");
  await expect(panel).toHaveAttribute("data-generation", "2");

  // The starts map: a filled cell for every (generation, start) that somebody finished.
  await expect(page.getByTestId("evolution-starts")).toHaveAttribute("data-starts", "10");
  const filled = every.start_finish_share.flat().filter((s: number) => s > 0).length;
  await expect(page.getByTestId("evolution-start-cell")).toHaveCount(filled);

  // The outcome grid: 50 networks x 10 starts, each cell the outcome the server sent.
  const outcomes = page.getByTestId("evolution-outcomes");
  await expect(outcomes).toHaveAttribute("data-generation", "2");
  const cells = page.getByTestId("evolution-outcome-cell");
  await expect(cells).toHaveCount(500);
  const finished = await cells.evaluateAll((els) => els.filter((e) => e.getAttribute("data-status") === "finished").length);
  const sent = grid.status.flat().filter((s: string) => s === "finished").length;
  expect(finished).toBe(sent);
  // the best network sits in the first column
  const first = page.locator('[data-testid="evolution-outcome-cell"][data-rank="1"][data-start="0"]');
  await expect(first).toHaveAttribute("data-row", String(grid.order[0]));

  // Picking start 2 redraws the fitness charts with f on that start.
  await page.getByTestId("evolution-start").selectOption("1");
  await expect(panel).toHaveAttribute("data-start", "1");
  await expect(page.getByTestId("evolution-fitness")).toContainText("Fitness f on start 2");
  await expect(page.getByTestId("evolution-generation")).toContainText("Generation 2 on start 2: best");
  await expect(page.getByTestId("evolution-start-row").first()).toBeVisible();

  // Hover a cell for its drive in words, then click it: generation 2 races from start 4 with that car selected.
  const cell = page.locator('[data-testid="evolution-outcome-cell"][data-rank="3"][data-start="3"]');
  const row = Number(await cell.getAttribute("data-row"));
  await cell.hover();
  const hover = page.getByTestId("evolution-outcomes-hover");
  await expect(hover).toContainText("rank 3");
  await expect(hover).toContainText("start 4:");
  await expect(hover).toContainText(grid.status[row][3] === "timeout" ? "out of time" : grid.status[row][3]);
  await cell.click();
  const canvas = page.getByTestId("environment-canvas").first();
  await expect(canvas).toHaveAttribute("data-race", "2:3:1"); // generation 2, start 4, ghost = generation 1
  await expect(page.getByTestId("select-car").first()).toHaveValue(String(row));
  await expect(page.getByTestId("hud-race-role")).toContainText(`car ${row + 1} · rank 3`);
  await expect(page.getByTestId("evolution-raced-row")).toBeVisible();
});
