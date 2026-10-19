import { expect, type Page, test } from "@playwright/test";

// The multilayer Network view (task 2.7). Every test starts a fresh page, which opens a fresh live
// session on the server. The random-networks driver gives every car its own [7, 16, 16, 2] network.

async function useRandomNetworks(page: Page, cars = 20) {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-policy").selectOption("random_networks");
  await expect(page.getByTestId("select-policy")).toHaveValue("random_networks");
  const box = page.getByTestId("input-cars");
  await box.fill(String(cars));
  await box.press("Enter");
  await expect(page.getByTestId("select-car").locator("option")).toHaveCount(cars);
  await expect(page.getByTestId("hud-step")).toHaveText("0");
}

async function goToNetwork(page: Page) {
  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByTestId("network-diagram")).toBeVisible();
}

async function chooseLayout(page: Page, name: "Page" | "Split" | "All") {
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name }).click();
}

const nodeValues = (page: Page) =>
  page
    .locator('[data-testid^="net-node-"]')
    .evaluateAll((nodes) => nodes.map((n) => n.getAttribute("data-activation")).join(","));

test("the Network page draws the selected car's network: 4 columns, 41 neurons, 400 weights", async ({
  page,
}) => {
  await useRandomNetworks(page);
  await goToNetwork(page);

  await expect(page.getByTestId("network-view")).toBeVisible();
  await expect(page.locator('[data-testid^="net-node-"]')).toHaveCount(7 + 16 + 16 + 2);
  await expect(page.locator('[data-testid^="net-edge-"]')).toHaveCount(7 * 16 + 16 * 16 + 16 * 2);
  for (const title of ["Inputs (7)", "Hidden 1 (16)", "Hidden 2 (16)", "Outputs (2)"]) {
    await expect(page.getByTestId("network-diagram").getByText(title)).toBeVisible();
  }
  await expect(page.getByText("The network of car 1")).toBeVisible();
  await expect(page.getByText("7 → 16 → 16 → 2 neurons, 434 numbers")).toBeVisible();
  await expect(page.getByTestId("network-legend")).toContainText("positive weight");
  await expect(page.getByTestId("network-legend")).toContainText("negative weight");
});

test("weights are drawn with a sign colour and a thickness that grows with the size", async ({
  page,
}) => {
  await useRandomNetworks(page);
  await goToNetwork(page);
  const edges = await page.locator('[data-testid^="net-edge-"]').evaluateAll((lines) =>
    lines.map((line) => ({
      weight: Number(line.getAttribute("data-weight")),
      width: Number(line.getAttribute("stroke-width")),
      sign: line.classList.contains("net-edge-neg") ? "neg" : "pos",
    })),
  );
  expect(edges.some((e) => e.weight > 0) && edges.some((e) => e.weight < 0)).toBe(true);
  for (const e of edges) expect(e.sign).toBe(e.weight < 0 ? "neg" : "pos");
  const biggest = edges.reduce((a, b) => (Math.abs(a.weight) > Math.abs(b.weight) ? a : b));
  const smallest = edges.reduce((a, b) => (Math.abs(a.weight) < Math.abs(b.weight) ? a : b));
  expect(biggest.width).toBeGreaterThan(smallest.width);
});

test("the input nodes show the same numbers as the Environment view's rays", async ({ page }) => {
  await useRandomNetworks(page);
  await chooseLayout(page, "All"); // Environment and Network side by side
  await expect(page.getByTestId("network-diagram").first()).toBeVisible();
  await page.getByTestId("select-car").selectOption("4");
  await expect(page.getByText("The network of car 5").first()).toBeVisible();
  for (let i = 0; i < 6; i++) await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("hud-step")).toHaveText("6");

  const read = async (testId: string) =>
    Number(await page.getByTestId(testId).first().getAttribute("data-value"));
  for (const k of [0, 1, 2, 3, 4]) {
    const metres = await read(`ray-${k}`);
    const input = await read(`net-input-${k}`);
    expect(input).toBeCloseTo(metres / 50, 2); // the rays are metres, the inputs metres / 50
  }
});

test("the activations update as the car drives", async ({ page }) => {
  await useRandomNetworks(page);
  await chooseLayout(page, "All");
  await page.getByTestId("select-car").selectOption("4"); // a seed-0 brain that moves
  await expect(page.getByText("The network of car 5").first()).toBeVisible();
  const before = await nodeValues(page);
  for (let i = 0; i < 8; i++) await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("hud-step")).toHaveText("8");
  await expect.poll(() => nodeValues(page)).not.toBe(before);
});

test("choosing another car shows that car's own network", async ({ page }) => {
  await useRandomNetworks(page);
  await chooseLayout(page, "All");
  await expect(page.getByTestId("network-diagram").first()).toBeVisible();
  const weight = () => page.getByTestId("net-edge-0-0-0").first().getAttribute("data-weight");
  const first = await weight();
  await page.getByTestId("select-car").selectOption("7");
  await expect(page.getByText("The network of car 8").first()).toBeVisible();
  await expect.poll(weight).not.toBe(first);
  await page.getByTestId("select-car").selectOption("0");
  await expect(page.getByText("The network of car 1").first()).toBeVisible();
  await expect.poll(weight).toBe(first); // car 1's weights again, exactly
});

test("hovering a circle or a line shows its exact values", async ({ page }) => {
  await useRandomNetworks(page);
  await goToNetwork(page);
  const readout = page.getByTestId("network-readout");
  await expect(readout).toContainText("Hover over a circle or a line");

  await page.getByTestId("net-node-3-0").hover();
  await expect(readout).toContainText("Steer: z = ");
  await expect(readout).toContainText("tanh(z) = ");

  await page.getByTestId("net-node-0-5").hover();
  await expect(readout).toContainText("Speed: value ");

  // A line: its tooltip text holds the exact weight (4 decimals) and the two neurons it joins.
  const edge = page.getByTestId("net-edge-2-3-1"); // Hidden 2, neuron 4 -> Throttle
  const weight = Number(await edge.getAttribute("data-weight"));
  const text = (await edge.locator("title").textContent()) ?? "";
  expect(text).toContain("Hidden 2, neuron 4 → Throttle: weight ");
  expect(text).toContain(`${weight < 0 ? "−" : ""}${Math.abs(weight).toFixed(4)}`);
  await edge.dispatchEvent("mouseover");
  await expect(readout).toContainText("Hidden 2, neuron 4 → Throttle: weight ");
});

test("the compact network also shows in the Environment page's drawer", async ({ page }) => {
  await useRandomNetworks(page);
  const compact = page.getByTestId("network-view-compact");
  await expect(compact).toBeVisible();
  await expect(compact.locator('[data-testid^="net-node-"]')).toHaveCount(41);
});

test("the other drivers still get the Neuron view", async ({ page }) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-policy").selectOption("neuron");
  await expect(page.getByTestId("select-policy")).toHaveValue("neuron");
  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByTestId("neuron-view")).toBeVisible();
  await expect(page.getByTestId("network-diagram")).toHaveCount(0);
});
