import { expect, test } from "@playwright/test";

// The dataset inspector (task 6.2, ADR-045): the Datasets tab shows a demonstration dataset's
// label and observation histograms and its frames on the track map, all numbers as the server
// computed them. global-setup.ts records an expert dataset tagged "e2e" (the exam's 10 starts)
// and one tagged "e2ecmp" (the same plus 2 random starts) for the comparator (6.2 follow-up).

interface Summary {
  dataset_id: string;
  demo_frames: number;
}

test("the Datasets tab shows the expert dataset's histograms and map", async ({ page }) => {
  const { datasets } = (await (await page.request.get("/api/datasets")).json()) as { datasets: Summary[] };
  const dataset = datasets.find((d) => d.dataset_id.endsWith("-expert-e2e"));
  expect(dataset, "an e2e expert dataset").toBeTruthy();
  const detail = await (await page.request.get(`/api/datasets/${dataset!.dataset_id}`)).json();

  await page.goto("/#/experiment");
  await page.getByTestId("experiment-tab-datasets").click();
  await expect(page.getByTestId("dataset-select")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("dataset-select").selectOption(dataset!.dataset_id);

  // The header and the 2 + 7 histograms, with the server's covered ranges and dense coverage.
  await expect(page.getByTestId("dataset-summary")).toHaveAttribute("data-frames", String(dataset!.demo_frames));
  const histograms = page.getByTestId("dataset-histogram");
  await expect(histograms).toHaveCount(9);
  await expect(histograms.first()).toHaveAttribute("data-name", "steer");
  const heading = page.locator('[data-testid="dataset-histogram"][data-name="heading_error"]');
  await expect(heading.getByTestId("dataset-covered")).toHaveAttribute("data-covered", String(detail.observations[6].covered));
  // 6.2 follow-up: every histogram has a dense line and the 1 % line, and the thick stretches
  // drawn add up to the server's dense coverage d.
  for (const [i, h] of [...detail.labels, ...detail.observations].entries()) {
    const figure = histograms.nth(i);
    await expect(figure.getByTestId("dataset-covered")).toHaveAttribute("data-dense", String(h.dense));
    await expect(figure.getByTestId("dataset-dense").first()).toBeAttached();
    await expect(figure.getByTestId("dataset-threshold")).toBeAttached();
    const drawn = Number(await figure.getAttribute("data-dense-drawn"));
    expect(Math.abs(drawn - h.dense), h.name).toBeLessThan(1e-5);
  }
  const leftRay = page.locator('[data-testid="dataset-histogram"][data-name="ray_left"]');
  await expect(leftRay.getByTestId("dataset-covered")).toContainText("densely covers");
  await expect(leftRay.getByTestId("dataset-covered")).toContainText("reaches");

  // Hovering a bar reads out its bin.
  await histograms.first().locator(".toy-hit").nth(9).hover();
  await expect(page.getByTestId("dataset-hist-hover")).toContainText("frames");

  // The map: every k-th frame on the track, coloured by steer, then by throttle.
  const map = page.getByTestId("dataset-map");
  await expect(map).toHaveAttribute("data-points", String(detail.map.x.length));
  await expect(map).toHaveAttribute("data-colour", "steer");
  await expect(page.getByTestId("dataset-map-legend")).toContainText("steer left");
  await page.getByTestId("dataset-colour-throttle").click();
  await expect(map).toHaveAttribute("data-colour", "throttle");
  await expect(page.getByTestId("dataset-map-legend")).toContainText("full throttle");

  // The episodes table: the exam's 10 starts, all finished.
  await expect(page.getByTestId("dataset-episodes-row")).toHaveCount(10);
});

test("Compare with shows a second dataset in every chart and on the map", async ({ page }) => {
  const { datasets } = (await (await page.request.get("/api/datasets")).json()) as { datasets: Summary[] };
  const a = datasets.find((d) => d.dataset_id.endsWith("-expert-e2e"));
  const b = datasets.find((d) => d.dataset_id.endsWith("-expert-e2ecmp"));
  expect(a && b, "the two e2e expert datasets").toBeTruthy();
  const detailA = await (await page.request.get(`/api/datasets/${a!.dataset_id}`)).json();
  const detailB = await (await page.request.get(`/api/datasets/${b!.dataset_id}`)).json();

  await page.goto("/#/experiment");
  await page.getByTestId("experiment-tab-datasets").click();
  await expect(page.getByTestId("dataset-select")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("dataset-select").selectOption(a!.dataset_id);
  await expect(page.getByTestId("dataset-summary")).toHaveAttribute("data-frames", String(a!.demo_frames));
  await expect(page.getByTestId("dataset-summary-b")).toHaveCount(0);

  await page.getByTestId("dataset-compare").selectOption(b!.dataset_id);
  // Two header lines, two episode tables.
  await expect(page.getByTestId("dataset-summary-b")).toHaveAttribute("data-frames", String(b!.demo_frames));
  await expect(page.getByTestId("dataset-episodes-b-row")).toHaveCount(detailB.meta.episodes.length);

  // Every histogram: B's bars (one per non-empty bin of B) and both dense numbers as the server sent them.
  const histograms = page.getByTestId("dataset-histogram");
  for (const [i, h] of [...detailB.labels, ...detailB.observations].entries()) {
    const figure = histograms.nth(i);
    const nonEmpty = h.counts.filter((c: number) => c > 0).length;
    await expect(figure).toHaveAttribute("data-bars-b", String(nonEmpty));
    await expect(figure.locator("g.ds-side-b rect.ds-bar")).toHaveCount(nonEmpty);
    await expect(figure.getByTestId("dataset-covered")).toHaveAttribute("data-dense-b", String(h.dense));
  }
  // The steer bin [0, 0.1) is filled by both: two bars in it, and the hover reads out both.
  const steer = histograms.first();
  await expect(steer.locator('rect.ds-bar[data-bin="10"]')).toHaveCount(2);
  await steer.locator(".toy-hit").nth(10).hover();
  await expect(page.getByTestId("dataset-hist-hover").locator("div")).toHaveCount(3);

  // The map: both datasets on the shared track, coloured by dataset.
  const map = page.getByTestId("dataset-map");
  await expect(map).toHaveAttribute("data-points", String(detailA.map.x.length));
  await expect(map).toHaveAttribute("data-points-b", String(detailB.map.x.length));
  await expect(map).toHaveAttribute("data-colour", "dataset");
  await page.getByTestId("dataset-colour-steer").click();
  await expect(page.getByTestId("dataset-map-legend")).toContainText("rings = B");

  // "nothing" goes back to one dataset.
  await page.getByTestId("dataset-compare").selectOption("");
  await expect(page.getByTestId("dataset-summary-b")).toHaveCount(0);
  await expect(map).toHaveAttribute("data-points-b", "0");
  await expect(page.getByTestId("dataset-colour-dataset")).toHaveCount(0);
  await expect(histograms.first().locator("g.ds-side-b rect")).toHaveCount(0);
});
