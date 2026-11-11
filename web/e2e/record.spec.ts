import { expect, test } from "@playwright/test";

// Keyboard demonstrations (task 6.1b, ADR-051): press Record, drive with the keys, Save; the
// server writes dataset v1 with teacher "keyboard" into its datasets folder, and the Datasets tab
// lists it. The dataset lands in data/demos/ (git-ignored), tagged "e2ekey".

interface Summary {
  dataset_id: string;
  teacher: string;
  demo_frames: number;
  episodes: number;
}

test("Record, drive, Save: the dataset appears in the Datasets tab", async ({ page }) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");

  // Recording needs the keyboard driver and one car, so the button is there by default.
  await page.getByTestId("btn-record").click();
  await expect(page.getByTestId("rec-badge")).toHaveAttribute("data-frames", "0");
  // The scenario cannot change under a recording.
  await expect(page.getByTestId("select-policy")).toBeDisabled();

  // Drive: hold the up arrow while the simulation plays.
  await page.getByTestId("btn-play").click();
  await page.keyboard.down("ArrowUp");
  await expect.poll(async () => Number(await page.getByTestId("rec-badge").getAttribute("data-frames"))).toBeGreaterThan(20);
  await page.keyboard.up("ArrowUp");
  await page.getByTestId("btn-play").click(); // pause

  await page.getByTestId("record-tag").fill("e2ekey");
  await page.getByTestId("btn-record-save").click();
  const saved = page.getByTestId("record-saved");
  await expect(saved).toBeVisible();
  const id = (await saved.getAttribute("data-dataset"))!;
  expect(id).toMatch(/-keyboard-e2ekey/);
  await expect(page.getByTestId("rec-badge")).toHaveCount(0);
  await expect(page.getByTestId("select-policy")).toBeEnabled();

  const { datasets } = (await (await page.request.get("/api/datasets")).json()) as { datasets: Summary[] };
  const found = datasets.find((d) => d.dataset_id === id);
  expect(found?.teacher).toBe("keyboard");
  expect(found!.demo_frames).toBeGreaterThan(20);
  expect(found!.episodes).toBe(1);
  // What the server saved is what the button counted.
  await expect(saved).toContainText(`${found!.demo_frames} frames`);

  await page.goto("/#/experiment");
  await page.getByTestId("experiment-tab-datasets").click();
  await expect(page.getByTestId("dataset-select")).toBeEnabled({ timeout: 15_000 });
  await page.getByTestId("dataset-select").selectOption(id);
  await expect(page.getByTestId("dataset-summary")).toHaveAttribute("data-frames", String(found!.demo_frames));
  // The keyboard gives full throttle: all frames in the last throttle bin.
  const throttle = page.locator('[data-testid="dataset-histogram"][data-name="throttle"]');
  await expect(throttle).toBeVisible();
});

test("Discard leaves no dataset, and a second car refuses the recording", async ({ page }) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  const before = ((await (await page.request.get("/api/datasets")).json()) as { datasets: Summary[] }).datasets.length;

  await page.getByTestId("btn-record").click();
  await expect(page.getByTestId("rec-badge")).toBeVisible();
  await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("rec-badge")).toHaveAttribute("data-frames", "1");
  await page.getByTestId("btn-record-discard").click();
  await expect(page.getByTestId("rec-badge")).toHaveCount(0);
  const after = ((await (await page.request.get("/api/datasets")).json()) as { datasets: Summary[] }).datasets.length;
  expect(after).toBe(before);

  // Two cars: the server says why it refuses.
  await page.getByTestId("input-cars").fill("2");
  await page.getByTestId("input-cars").press("Enter");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("btn-record").click();
  await expect(page.getByText("exactly one car")).toBeVisible();
  await expect(page.getByTestId("rec-badge")).toHaveCount(0);
});
