import { expect, type Locator, type Page, test } from "@playwright/test";

// Selection linking (task 2.8): one shared selection drives every view. Clicking a car on the
// track, stepping through cars on the Network page, and clicking neurons or weights all change
// the same selection, and every view follows it.

async function useRandomNetworks(page: Page, cars = 20, randomStart = false) {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-policy").selectOption("random_networks");
  await expect(page.getByTestId("select-policy")).toHaveValue("random_networks");
  const box = page.getByTestId("input-cars");
  await box.fill(String(cars));
  await box.press("Enter");
  await expect(page.getByTestId("select-car").locator("option")).toHaveCount(cars);
  if (randomStart) {
    // spread the cars round the track, so each one can be clicked on its own
    await page.getByTestId("check-random-start").click();
    await expect(page.getByTestId("check-random-start")).toBeChecked();
  }
  await expect(page.getByTestId("hud-step")).toHaveText("0");
}

async function chooseLayout(page: Page, name: "Page" | "Split" | "All") {
  await page.getByRole("group", { name: "Layout" }).getByRole("button", { name }).click();
}

/** Screen positions of every car on a canvas, as the view publishes them. */
async function carPositions(canvas: Locator): Promise<[number, number][]> {
  const text = (await canvas.getAttribute("data-cars")) ?? "";
  return text === "" ? [] : text.split(",").map((p) => p.split(":").map(Number) as [number, number]);
}

test("clicking a car on the track switches the network diagram to that car", async ({ page }) => {
  await useRandomNetworks(page, 20, true);
  await chooseLayout(page, "All"); // track and network side by side
  await page.getByTestId("camera-fit").click();
  const canvas = page.getByTestId("environment-canvas").first();
  await expect(page.getByText("The network of car 1").first()).toBeVisible();

  // Pick a car that has no other car close to it on screen, so the click is unambiguous.
  await expect.poll(async () => (await carPositions(canvas)).length).toBe(20);
  const cars = await carPositions(canvas);
  const alone = cars.findIndex(
    ([x, y], i) =>
      i > 0 &&
      cars.every(([u, v], j) => j === i || Math.hypot(u - x, v - y) > 30) &&
      x > 5 &&
      y > 5,
  );
  expect(alone).toBeGreaterThan(0);
  const before = await page.getByTestId("net-edge-0-0-0").first().getAttribute("data-weight");

  await canvas.click({ position: { x: cars[alone][0], y: cars[alone][1] } });
  await expect(page.getByText(`The network of car ${alone + 1}`).first()).toBeVisible();
  await expect(page.getByTestId("select-car")).toHaveValue(String(alone));
  await expect(page.getByTestId("car-label").first()).toHaveText(`Car ${alone + 1} of 20`);
  await expect
    .poll(() => page.getByTestId("net-edge-0-0-0").first().getAttribute("data-weight"))
    .not.toBe(before);
});

test("the Network page's car switcher changes the Environment view's selected car", async ({
  page,
}) => {
  await useRandomNetworks(page);
  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByTestId("car-label").first()).toHaveText("Car 1 of 20");
  await page.getByTestId("car-next").first().click();
  await page.getByTestId("car-next").first().click();
  await expect(page.getByTestId("car-label").first()).toHaveText("Car 3 of 20");
  await expect(page.getByText("The network of car 3")).toBeVisible();
  await page.getByTestId("car-prev").first().click();
  await page.getByTestId("car-prev").first().click();
  await page.getByTestId("car-prev").first().click(); // wraps round to the last car
  await expect(page.getByTestId("car-label").first()).toHaveText("Car 20 of 20");

  await page.getByRole("link", { name: "Environment" }).click();
  await expect(page.getByTestId("select-car")).toHaveValue("19");
});

test("selecting a ray input or a weight leaving it highlights that ray on the track", async ({
  page,
}) => {
  await useRandomNetworks(page);
  await page.getByRole("link", { name: "Network" }).click();
  const node = page.getByTestId("net-node-0-1"); // Ray +30 degrees
  await node.click();
  await expect(node).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("network-readout")).toContainText("Selected: Ray +30°: value ");

  const highlighted = () =>
    page.getByTestId("environment-canvas").first().getAttribute("data-highlighted-rays");
  await page.getByRole("link", { name: "Environment" }).click();
  await expect.poll(highlighted).toBe("1");

  // a weight from the far-right ray (input 4) to hidden neuron 3
  await page.getByRole("link", { name: "Network" }).click();
  await page.getByTestId("net-hit-0-4-2").dispatchEvent("click");
  await expect(page.getByTestId("net-edge-0-4-2")).toHaveAttribute("data-selected", "true");
  await expect(node).toHaveAttribute("aria-pressed", "false"); // the weight replaced the neuron
  await expect(page.getByTestId("network-readout")).toContainText(
    "Selected: Ray −60° → Hidden 1, neuron 3: weight ",
  );
  await page.getByRole("link", { name: "Environment" }).click();
  await expect.poll(highlighted).toBe("4");

  // speed has no ray; clicking it again clears the selection
  await page.getByRole("link", { name: "Network" }).click();
  await page.getByTestId("net-node-0-5").click();
  await page.getByRole("link", { name: "Environment" }).click();
  await expect.poll(highlighted).toBe("");
  await page.getByRole("link", { name: "Network" }).click();
  await page.getByTestId("net-node-0-5").click();
  await expect(page.getByTestId("net-node-0-5")).toHaveAttribute("aria-pressed", "false");
  // still hovered: its values show, but no longer as the selection
  await expect(page.getByTestId("network-readout")).toHaveText(/^Speed: value /);
  await page.mouse.move(2, 2);
  await expect(page.getByTestId("network-readout")).toContainText("Hover over a circle");
});

test("the selected neuron stays selected when switching cars, and its values update", async ({
  page,
}) => {
  await useRandomNetworks(page);
  await page.getByRole("link", { name: "Network" }).click();
  await page.getByTestId("net-node-3-0").click(); // Steer
  const readout = page.getByTestId("network-readout");
  await expect(readout).toContainText("Selected: Steer: z = ");
  const first = await readout.innerText();
  await page.getByTestId("car-next").first().click();
  await expect(page.getByText("The network of car 2")).toBeVisible();
  await expect(page.getByTestId("net-node-3-0")).toHaveAttribute("aria-pressed", "true");
  await expect(readout).toContainText("Selected: Steer: z = ");
  await expect(readout).not.toHaveText(first); // car 2's steer neuron has its own values
});

test("clicking a cell in the drawer selects that neuron in the full diagram, and back", async ({
  page,
}) => {
  await useRandomNetworks(page);
  const drawer = page.getByTestId("network-view-compact");
  const cell = drawer.getByTestId("cnet-cell-1-5"); // Hidden 1, neuron 6
  await cell.click();
  await expect(cell).toHaveAttribute("aria-pressed", "true");
  await expect(drawer.getByTestId("network-compact-readout")).toContainText("Hidden 1, neuron 6: z = ");

  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByTestId("net-node-1-5")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("network-readout")).toContainText("Selected: Hidden 1, neuron 6");

  // and the other way: an output picked in the full diagram shows up in the drawer
  await page.getByTestId("net-node-3-1").click(); // Throttle
  await page.getByRole("link", { name: "Environment" }).click();
  await expect(page.getByTestId("cnet-output-1")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("cnet-cell-1-5")).toHaveAttribute("aria-pressed", "false");
});

test("loading a new scenario clears the neuron selection", async ({ page }) => {
  await useRandomNetworks(page, 5);
  await page.getByRole("link", { name: "Network" }).click();
  await page.getByTestId("net-node-2-2").click();
  await expect(page.getByTestId("net-node-2-2")).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("link", { name: "Environment" }).click();
  const box = page.getByTestId("input-cars");
  await box.fill("6");
  await box.press("Enter");
  await expect(page.getByTestId("select-car").locator("option")).toHaveCount(6);
  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByTestId("net-node-2-2")).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByTestId("car-label").first()).toHaveText("Car 1 of 6");
});
