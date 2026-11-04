import { expect, test } from "@playwright/test";

// The dataset inspector (task 6.2, ADR-045): the Datasets tab shows a demonstration dataset's
// label and observation histograms and its frames on the track map, all numbers as the server
// computed them. global-setup.ts records an expert dataset tagged "e2e" (the exam's 10 starts).

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

  // The header and the 2 + 7 histograms, with the server's covered ranges.
  await expect(page.getByTestId("dataset-summary")).toHaveAttribute("data-frames", String(dataset!.demo_frames));
  const histograms = page.getByTestId("dataset-histogram");
  await expect(histograms).toHaveCount(9);
  await expect(histograms.first()).toHaveAttribute("data-name", "steer");
  const heading = page.locator('[data-testid="dataset-histogram"][data-name="heading_error"]');
  await expect(heading.getByTestId("dataset-covered")).toHaveAttribute("data-covered", String(detail.observations[6].covered));

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
  await expect(page.getByTestId("dataset-episode-row")).toHaveCount(10);
});
