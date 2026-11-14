import { expect, test, type Page } from "@playwright/test";

// A GA run's lineage (task 7.8, ADR-060): the line of descent in the Timeline's Evolution panel,
// and in population mode the Network view's "colour lines by mutation since the parent".
// global-setup.ts trains a 2-generation GA run of 50 networks (tag "e2e"), with its lineage.

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

test("the line of descent races its ancestors, and a child's mutations colour its network", async ({ page }) => {
  test.setTimeout(60_000);
  const runId = await gaRun(page);
  const line = await (await page.request.get(`/api/runs/${runId}/generations/2/ancestry`)).json();
  expect(line.steps.map((s: { unit: number }) => s.unit)).toEqual([2, 1]);
  expect(line.steps[1].origin).toBe("random");
  // row 2 of generation 2 is a child (rows 0 and 1 are the elites)
  const mutation = await (await page.request.get(`/api/runs/${runId}/generations/2/rows/2/mutation`)).json();
  expect(mutation.origin).toBe("child");
  const mutatedWeights = (mutation.weight_mutated as boolean[][][]).flat(2).filter(Boolean).length;

  await page.goto("/#/experiment");
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name: "All" }).click();
  await page.getByTestId("experiment-tab-timeline").click();
  await expect(page.getByTestId("timeline-run")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("timeline-run").selectOption(runId);
  const slider = page.getByTestId("timeline-slider");
  await slider.fill((await slider.getAttribute("max")) ?? "0"); // generation 2

  // The champion's line: 2 steps, generation 1's a random network.
  const section = page.getByTestId("ancestry");
  await expect(section).toHaveAttribute("data-steps", "2");
  await expect(section).toHaveAttribute("data-row", String(line.row));
  await expect(page.getByTestId("ancestry-step-1")).toHaveAttribute("data-origin", "random");
  await expect(page.getByTestId("ancestry-step-1")).toHaveAttribute("data-row", String(line.steps[1].row));
  await expect(page.getByTestId("ancestry-summary")).toContainText("goes back 1 generation");
  const hit = page.getByTestId("ancestry-hit");
  const box = (await hit.boundingBox())!;
  await hit.hover({ position: { x: box.width * 0.1, y: box.height / 2 } });
  await expect(page.getByTestId("ancestry-hover")).toContainText("generation 1");
  await expect(page.getByTestId("ancestry-hover")).toContainText("a random genome");

  // Its generation-1 ancestor races with that car selected.
  await section.locator("summary").click();
  await page.getByTestId("ancestry-race-1").click();
  const canvas = page.getByTestId("environment-canvas").first();
  await expect(canvas).toHaveAttribute("data-race", /^1:0:/);
  await expect(page.getByTestId("select-car").first()).toHaveValue(String(line.steps[1].row));
  await expect(page.getByTestId("lineage-banner").first()).toContainText("a random genome of generation 1");

  // Another row's line: row 2 of generation 2, a child; race it and show its mutation.
  await page.getByTestId("ancestry-row").selectOption("2");
  await expect(section).toHaveAttribute("data-row", "2");
  await page.getByTestId("ancestry-race-2").click();
  await expect(canvas).toHaveAttribute("data-race", /^2:0:/);
  await expect(page.getByTestId("select-car").first()).toHaveValue("2");
  const banner = page.getByTestId("lineage-banner").first();
  await expect(banner).toHaveAttribute("data-origin", "child");
  await expect(banner).toContainText(`child of rows ${mutation.parents[0]} and ${mutation.parents[1]} of generation 1`);
  await page.getByTestId("show-mutation").first().check();
  const legend = page.getByTestId("network-mutation-legend").first();
  await expect(legend).toHaveAttribute("data-weights", String(mutatedWeights));
  await expect(page.locator('[data-testid^="net-edge-"][data-mutated="true"]')).toHaveCount(mutatedWeights);
});
