import { expect, test, type Page } from "@playwright/test";

import { formatNumber } from "../src/views/experiment/charts";

// "Compare runs" in the Experiment view (task 4.7b, ADR-033). global-setup.ts trains a 5-seed sine
// sweep tagged "e2esweep" (100 epochs); the newest such sweep is used here.

interface RunSummary {
  run_id: string;
  status: string;
  sweep?: string | null;
}

async function latestSweep(
  page: Page,
): Promise<{ sweep: string; runIds: string[] }> {
  const { runs } = (await (await page.request.get("/api/runs")).json()) as {
    runs: RunSummary[];
  };
  const sweepRuns = runs.filter(
    (r) => /-toy_sine-e2esweep-s\d+$/.test(r.run_id) && r.sweep,
  );
  expect(sweepRuns.length, "an e2e sweep").toBeGreaterThanOrEqual(5);
  const sweep = sweepRuns[0].sweep!; // newest first
  return {
    sweep,
    runIds: runs.filter((r) => r.sweep === sweep).map((r) => r.run_id),
  };
}

async function openCompare(page: Page) {
  await page.goto("/#/experiment");
  await page.getByTestId("experiment-tab-compare").click();
  await expect(page.getByTestId("compare")).toBeVisible();
  await expect(page.getByTestId("runlist-group").first()).toBeVisible();
}

test("compare runs: a 5-seed sweep shows one band, one median line and five run lines", async ({
  page,
}) => {
  const { sweep, runIds } = await latestSweep(page);
  expect(runIds).toHaveLength(5);
  await openCompare(page);

  const group = page.locator(
    `[data-testid="runlist-group"][data-group-id="${sweep}"]`,
  );
  await group.locator("input").first().check();

  await expect(page.getByTestId("compare-run-line")).toHaveCount(5);
  await expect(page.getByTestId("compare-median")).toHaveCount(1);
  await expect(page.getByTestId("compare-band")).toHaveCount(1);
  await expect(page.getByTestId("compare-band")).toHaveAttribute("d", /^M.+Z$/);

  // The table shows the server's numbers.
  const answer = await (
    await page.request.get(
      `/api/runs/compare?ids=${runIds.join(",")}&metric=loss`,
    )
  ).json();
  expect(answer.groups).toHaveLength(1);
  await expect(page.getByTestId("compare-final-median")).toHaveText(
    formatNumber(answer.groups[0].final_median),
  );

  // Hovering shows the read-out; unticking one seed leaves four run lines.
  await page.getByTestId("compare-chart").locator("rect.toy-hit").hover();
  await expect(page.getByTestId("compare-hover")).toContainText(
    "median [Q1, Q3]",
  );
  await page
    .locator(`[data-testid="runlist-run"][data-run-id="${runIds[0]}"]`)
    .uncheck();
  await expect(page.getByTestId("compare-run-line")).toHaveCount(4);
});

test("compare runs: validation loss is skipped for runs without a validation set", async ({
  page,
}) => {
  const { sweep } = await latestSweep(page);
  await openCompare(page);
  await page
    .locator(`[data-testid="runlist-group"][data-group-id="${sweep}"]`)
    .locator("input")
    .first()
    .check();
  await expect(page.getByTestId("compare-median")).toHaveCount(1);
  await page.getByRole("button", { name: "validation loss" }).click();
  await expect(page.getByTestId("compare-skipped")).toContainText(
    "5 ticked runs have no validation loss",
  );
  await expect(page.getByTestId("compare-chart")).toHaveCount(0);
});
