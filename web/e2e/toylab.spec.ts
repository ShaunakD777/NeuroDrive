import { expect, type Page, test } from "@playwright/test";

// The toy lab in the Experiment view (task 3.8). global-setup.ts trains two small toy runs tagged
// "e2e" (sine for 300 epochs, XOR for 1000); the newest of each is used here.

interface Toy {
  unit: number;
  loss: number;
  glow: number[][][];
}

async function latestRun(page: Page, task: string): Promise<string> {
  const response = await page.request.get("/api/runs");
  const { runs } = (await response.json()) as { runs: { run_id: string; status: string }[] };
  const run = runs.find((r) => r.run_id.endsWith(`-toy_${task}-e2e`) && r.status === "completed");
  expect(run, `an e2e ${task} run`).toBeTruthy();
  return run!.run_id;
}

async function openRun(page: Page, runId: string) {
  await page.goto("/#/experiment");
  await expect(page.getByTestId("toylab")).toBeVisible();
  await page.getByTestId("run-pick").selectOption(runId);
  await expect(page.getByTestId("toylab-summary")).toHaveAttribute("data-run-id", runId);
}

async function moveSlider(page: Page, index: number, unit: number) {
  await page.getByTestId("checkpoint-slider").fill(String(index));
  await expect(page.getByTestId("checkpoint-label")).toHaveText(new RegExp(`^epoch ${unit} · loss`));
  await expect(page.getByTestId("prediction-plot")).toContainText(`epoch ${unit},`);
}

test("toy lab: scrubbing the sine run shows the predictions improving", async ({ page }) => {
  const runId = await latestRun(page, "sine");
  const { units } = await (await page.request.get(`/api/runs/${runId}/checkpoints`)).json();
  await openRun(page, runId);

  // The loss curve and one dot per checkpoint; the run starts at its last checkpoint.
  await expect(page.getByTestId("loss-line")).toHaveAttribute("d", /^M.+L/);
  await expect(page.getByTestId("loss-checkpoint")).toHaveCount(units.length);
  const last = units.length - 1;
  await expect(page.getByTestId("loss-cursor")).toHaveAttribute("data-unit", String(units[last]));

  await moveSlider(page, 0, 0);
  const plot = page.getByTestId("prediction-plot");
  const loss0 = Number(await plot.getAttribute("data-loss"));
  const curve0 = await page.getByTestId("prediction-curve").getAttribute("d");
  await expect(page.getByTestId("loss-cursor")).toHaveAttribute("data-unit", "0");

  await moveSlider(page, last, units[last]);
  const lossEnd = Number(await plot.getAttribute("data-loss"));
  const curveEnd = await page.getByTestId("prediction-curve").getAttribute("d");
  expect(curveEnd).not.toBe(curve0);
  expect(lossEnd).toBeLessThan(loss0 / 5); // the prediction got much closer to the data

  // Every step in between: the loss the plot reports never gets worse than at epoch 0.
  for (let i = 1; i < last; i++) {
    await moveSlider(page, i, units[i]);
    expect(Number(await plot.getAttribute("data-loss"))).toBeLessThan(loss0);
  }

  // Clicking the first checkpoint dot on the loss curve selects epoch 0 again.
  await page.getByTestId("loss-checkpoint").first().click();
  await expect(page.getByTestId("checkpoint-label")).toHaveText(/^epoch 0 · loss/);
});

test("toy lab: the network glows by the server's gradient sizes", async ({ page }) => {
  const runId = await latestRun(page, "sine");
  await openRun(page, runId);
  await moveSlider(page, 2, 2);
  const toy = (await (await page.request.get(`/api/runs/${runId}/checkpoints/2/toy`)).json()) as Toy;

  const edges = page.getByTestId("toy-edge");
  await expect(edges).toHaveCount(16 + 16); // [1, 16, 1]: 16 weights in, 16 out
  const drawn = await edges.evaluateAll((lines) =>
    lines.map((l) => ({
      name: l.getAttribute("data-name")!,
      glow: Number(l.getAttribute("data-glow")),
      opacity: Number(l.getAttribute("stroke-opacity")),
    })),
  );
  for (const e of drawn) {
    const [, layer, to, from] = /^W(\d+)\[(\d+),(\d+)\]$/.exec(e.name)!.map(Number);
    const glow = toy.glow[layer - 1][to][from];
    expect(e.glow).toBe(glow);
    expect(e.opacity).toBeCloseTo(0.08 + 0.92 * glow, 6); // charts.ts glowStroke
  }
  expect(Math.max(...drawn.map((e) => e.glow))).toBe(1);

  // Hovering a connection reads out its weight and gradient.
  await page.locator(".toy-edge-hit").first().hover();
  await expect(page.getByTestId("toy-network-readout")).toContainText(/W\d\[\d+,\d+\]: weight .* gradient/);
});

test("toy lab: the XOR run shows a map over the plane", async ({ page }) => {
  const runId = await latestRun(page, "xor");
  const { units } = await (await page.request.get(`/api/runs/${runId}/checkpoints`)).json();
  await openRun(page, runId);
  await moveSlider(page, units.length - 1, units[units.length - 1]);
  await expect(page.getByTestId("prediction-map").locator("rect")).toHaveCount(41 * 41);
  await expect(page.locator(".toy-point")).toHaveCount(4);
  await expect(page.locator(".toy-point-pos")).toHaveCount(2); // the two corners labelled 1
  expect(Number(await page.getByTestId("prediction-plot").getAttribute("data-loss"))).toBeLessThan(0.01);
});
