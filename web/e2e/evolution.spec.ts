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
  await expect(page.getByTestId("evolution-race")).toHaveText("Generation 1 races in the Environment view");
});
