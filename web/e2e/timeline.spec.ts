import { expect, test, type Page } from "@playwright/test";

// The run timeline (task 5.9, ADR-042): scrubbing a driving run's checkpoints changes all three
// views together, and the replayed drive is the exam drive. global-setup.ts trains a small
// random-search run (3 units, tag "e2e"); the newest completed one is used.

interface RunSummary {
  run_id: string;
  status: string;
}

const LABEL: Record<string, string> = {
  crashed: "Crashed",
  finished: "Finished",
  stalled: "Stalled",
  timeout: "Time limit",
};

async function searchRun(page: Page): Promise<{ runId: string; units: number[] }> {
  const { runs } = (await (await page.request.get("/api/runs")).json()) as { runs: RunSummary[] };
  const run = runs.find((r) => r.run_id.endsWith("-random_search-e2e") && r.status === "completed");
  expect(run, "an e2e random-search run").toBeTruthy();
  const { units } = await (await page.request.get(`/api/runs/${run!.run_id}/checkpoints`)).json();
  expect(units.length).toBeGreaterThanOrEqual(4);
  return { runId: run!.run_id, units };
}

/** The first weight of the checkpoint's network, as the server sends it (4 decimals). */
async function firstWeight(page: Page, runId: string, unit: number): Promise<number> {
  const network = await (await page.request.get(`/api/runs/${runId}/checkpoints/${unit}`)).json();
  return network.weights[0][0][0];
}

async function openTimeline(page: Page, runId: string) {
  await page.goto("/#/experiment");
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name: "All" }).click();
  await page.getByTestId("experiment-tab-timeline").click();
  await expect(page.getByTestId("timeline-run")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("timeline-run").selectOption(runId);
  await expect(page.getByTestId("timeline-slider")).toBeVisible();
}

async function expectAllViewsAt(page: Page, runId: string, unit: number) {
  // Experiment view: the cursor and the label.
  await expect(page.getByTestId("timeline-label")).toHaveAttribute("data-unit", String(unit));
  await expect(page.getByTestId("live-cursor")).toHaveAttribute("data-unit", String(unit));
  // Environment view: the checkpoint drives the exam's 10 starts.
  await expect(page.getByTestId("select-policy").first()).toHaveValue("checkpoint");
  await expect(page.getByTestId("select-policy").first().locator("option:checked")).toContainText(`unit ${unit}`);
  await expect(page.getByTestId("select-car").first().locator("option")).toHaveCount(10);
  // Network view: the checkpoint's own weights.
  await expect(page.getByTestId("replay-banner")).toHaveAttribute("data-unit", String(unit));
  await expect(page.getByTestId("net-edge-0-0-0")).toHaveAttribute("data-weight", String(await firstWeight(page, runId, unit)));
}

test("scrubbing the timeline changes all three views consistently", async ({ page }) => {
  const { runId, units } = await searchRun(page);
  await openTimeline(page, runId);

  await page.getByTestId("timeline-slider").fill("0");
  await expectAllViewsAt(page, runId, units[0]);
  // The first checkpoint has nothing to compare with: lines show the weights.
  await expect(page.getByTestId("show-change")).toBeDisabled();
  await expect(page.getByTestId("network-legend")).toBeVisible();

  // Checkpoint 1 holds a new network (random search draws afresh): every line shows a change.
  await page.getByTestId("timeline-slider").fill("1");
  await expectAllViewsAt(page, runId, units[1]);
  const first = await (await page.request.get(`/api/runs/${runId}/checkpoints/${units[1]}/diff?from=${units[0]}`)).json();
  expect(first.max_abs_delta).toBeGreaterThan(0);
  await expect(page.getByTestId("net-edge-0-0-0")).toHaveAttribute("data-change", String(first.weight_change[0][0][0]));
  await expect(page.getByTestId("network-change-legend")).toContainText(`since unit ${units[0]}`);

  const last = units.length - 1;
  await page.getByTestId("timeline-slider").fill(String(last));
  await expectAllViewsAt(page, runId, units[last]);
  // Later checkpoints: the lines show what changed since the previous checkpoint.
  await expect(page.getByTestId("show-change")).toBeChecked();
  await expect(page.getByTestId("network-change-legend")).toContainText(`since unit ${units[last - 1]}`);
  const diff = await (
    await page.request.get(`/api/runs/${runId}/checkpoints/${units[last]}/diff?from=${units[last - 1]}`)
  ).json();
  await expect(page.getByTestId("net-edge-0-0-0")).toHaveAttribute("data-change", String(diff.weight_change[0][0][0]));
  await page.getByTestId("show-change").uncheck();
  await expect(page.getByTestId("network-legend")).toBeVisible();
  await page.getByTestId("show-change").check();

  // Another track: the same checkpoint is replayed there.
  await page.getByTestId("timeline-track").selectOption("hairpins_01");
  await expect(page.getByTestId("select-track").first()).toHaveValue("hairpins_01");
  await expect(page.getByTestId("replay-banner")).toHaveAttribute("data-unit", String(units[last]));
});

test("a replayed checkpoint drives exactly as its exam card says", async ({ page }) => {
  test.setTimeout(120_000);
  const { runId, units } = await searchRun(page);
  await openTimeline(page, runId);
  const unit = units[units.length - 1];
  await page.getByTestId("timeline-replay").click();
  await expect(page.getByTestId("replay-banner")).toHaveAttribute("data-unit", String(unit));
  const rows = page.getByTestId("timeline-card-row");
  await expect(rows).toHaveCount(10);

  await page.getByTestId("select-speed").first().selectOption("8");
  await page.getByTestId("btn-play").first().click();
  await expect(page.getByTestId("btn-play").first()).toBeDisabled({ timeout: 90_000 }); // every car stopped

  // Each car's ending in the HUD is the one on the card.
  for (let car = 0; car < 10; car++) {
    const status = (await rows.nth(car).getAttribute("data-status"))!;
    await rows.nth(car).click();
    await expect(page.getByTestId("select-car").first()).toHaveValue(String(car));
    await expect(page.getByTestId("hud-status").first()).toHaveText(LABEL[status]);
  }
});
