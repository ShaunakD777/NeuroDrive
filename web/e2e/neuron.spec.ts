import { expect, type Page, test } from "@playwright/test";

// The Neuron driver and the Neuron view (task 1.9). Every test starts a fresh page, which opens a
// fresh live session on the server, so tests do not affect each other.

async function useNeuronDriver(page: Page) {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-policy").selectOption("neuron");
  await expect(page.getByTestId("select-policy")).toHaveValue("neuron");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
}

async function goToNetwork(page: Page) {
  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByTestId("neuron-view")).toBeVisible();
}

async function setNumber(page: Page, name: string, value: number) {
  const box = page.getByTestId(`number-${name}`);
  await box.fill(String(value));
  await box.press("Enter");
}

test("without the neuron driver the Network page offers to switch to it", async ({ page }) => {
  await page.goto("/#/network");
  await expect(page.getByTestId("neuron-empty")).toBeVisible();
  await page.getByTestId("use-neuron").click();
  await expect(page.getByTestId("neuron-z")).toBeVisible();
  await expect(page.getByTestId("slider-w_left")).toBeVisible();
});

test("the Neuron view shows the inputs, weights, sum, activation curve and steering dial", async ({
  page,
}) => {
  await useNeuronDriver(page);
  await goToNetwork(page);

  await expect(page.getByTestId("neuron-z")).toHaveText("0.00");
  await expect(page.getByTestId("neuron-output")).toHaveText("0.00");
  await expect(page.getByTestId("neuron-steer-label")).toHaveText("straight");
  await expect(page.getByTestId("select-activation")).toHaveValue("tanh");
  for (const name of ["w_left", "w_right", "bias", "throttle"]) {
    await expect(page.getByTestId(`slider-${name}`)).toBeVisible();
  }
  await expect(page.getByTestId("curve-dot")).toBeVisible();
  await expect(page.getByTestId("steer-dial")).toBeVisible();
  await expect(page.getByTestId("neuron-input-0")).not.toHaveText("");
  await expect(page.getByTestId("neuron-range")).toContainText("tanh");
});

test("changing a weight changes the steering in the next frame", async ({ page }) => {
  await useNeuronDriver(page);
  await goToNetwork(page);

  // With both weights 0 the sum is just the bias: z = 1, tanh(1) = 0.76 (a hard right turn).
  await setNumber(page, "bias", 1);
  await expect(page.getByTestId("neuron-z")).toHaveText("1.00");
  await expect(page.getByTestId("neuron-output")).toHaveText("0.76");
  await expect(page.getByTestId("neuron-steer-label")).toHaveText("hard right");

  // The Environment page's HUD shows the steering actually applied after the next step.
  await page.getByRole("link", { name: "Environment" }).click();
  await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("hud-steer")).toHaveText("0.76");

  await page.getByRole("link", { name: "Network" }).click();
  await setNumber(page, "bias", -1);
  await expect(page.getByTestId("neuron-output")).toHaveText("−0.76");
  await expect(page.getByTestId("neuron-steer-label")).toHaveText("hard left");
  await page.getByRole("link", { name: "Environment" }).click();
  await page.getByTestId("btn-step").click();
  await expect(page.getByTestId("hud-steer")).toHaveText("-0.76");
});

test("the weighted sum is the contributions plus the bias, as the server computed it", async ({
  page,
}) => {
  await useNeuronDriver(page);
  await goToNetwork(page);
  await setNumber(page, "w_left", -4);
  await setNumber(page, "w_right", 4);
  await setNumber(page, "bias", 0.1);
  await expect(page.getByTestId("number-w_right")).toHaveValue("4.00");

  const text = async (id: string) => (await page.getByTestId(id).innerText()).replace("−", "-");
  const x0 = Number(await text("neuron-input-0"));
  const x1 = Number(await text("neuron-input-1"));
  const c0 = Number(await text("neuron-contribution-0"));
  const c1 = Number(await text("neuron-contribution-1"));
  const z = Number(await text("neuron-z"));
  expect(c0).toBeCloseTo(-4 * x0, 1);
  expect(c1).toBeCloseTo(4 * x1, 1);
  expect(z).toBeCloseTo(c0 + c1 + 0.1, 1);
});

test("sigmoid can be chosen and its output is never negative", async ({ page }) => {
  await useNeuronDriver(page);
  await goToNetwork(page);

  await page.getByTestId("select-activation").selectOption("sigmoid");
  await expect(page.getByTestId("select-activation")).toHaveValue("sigmoid");
  await expect(page.getByTestId("neuron-range")).toContainText("never negative");
  // z = bias = -1: tanh would give -0.76 (left), sigmoid gives 0.27, still positive.
  await setNumber(page, "bias", -1);
  await expect(page.getByTestId("neuron-output")).toHaveText("0.27");
  await setNumber(page, "bias", -20);
  await expect(page.getByTestId("neuron-output")).toHaveText("0.00");
  await expect(page.getByTestId("neuron-steer-label")).toHaveText("straight");

  await page.getByTestId("select-activation").selectOption("relu");
  await expect(page.getByTestId("neuron-output")).toHaveText("0.00"); // relu(-20) = 0
  await page.getByTestId("select-activation").selectOption("linear");
  await expect(page.getByTestId("neuron-output")).toHaveText("−20.00");
});

test("the slider moves the weight and the car, not just the display", async ({ page }) => {
  await useNeuronDriver(page);
  await goToNetwork(page);
  await page.getByTestId("slider-bias").fill("0.5");
  await expect(page.getByTestId("neuron-z")).toHaveText("0.50");
  await expect(page.getByTestId("number-bias")).toHaveValue("0.50");
});

test("weights are behaviour: all zeros crash, opposite-sign weights keep driving", async ({
  page,
}) => {
  await useNeuronDriver(page);
  await page.getByTestId("select-speed").selectOption("8");

  // Zero weights: the neuron steers straight, so the car leaves the first bend.
  await page.getByTestId("btn-play").click();
  await expect(page.getByTestId("status-banner")).toContainText("Crashed", { timeout: 30_000 });

  // Opposite-sign weights steer away from the nearer wall.
  await goToNetwork(page);
  await setNumber(page, "w_left", -8);
  await setNumber(page, "w_right", 8);
  await page.getByRole("link", { name: "Environment" }).click();
  await page.getByTestId("btn-reset").click();
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("btn-play").click();
  await expect
    .poll(async () => Number((await page.getByTestId("hud-progress").innerText()).split(" ")[0]))
    .toBeGreaterThan(150);
  await expect(page.getByTestId("hud-status")).toHaveText("Running");
});

test("clicking the neuron highlights the two rays it reads on the track, and clicking again clears", async ({
  page,
}) => {
  await useNeuronDriver(page);
  await goToNetwork(page);
  const node = page.getByTestId("neuron-node");
  await expect(node).toHaveAttribute("aria-pressed", "false");

  await node.click();
  await expect(node).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("link", { name: "Environment" }).click();
  await expect(page.getByTestId("environment-canvas")).toHaveAttribute(
    "data-highlighted-rays",
    "0,4",
  );

  await page.getByRole("link", { name: "Network" }).click();
  await page.getByTestId("neuron-node").click();
  await expect(page.getByTestId("neuron-node")).toHaveAttribute("aria-pressed", "false");
  await page.getByRole("link", { name: "Environment" }).click();
  await expect(page.getByTestId("environment-canvas")).toHaveAttribute("data-highlighted-rays", "");
});

test("the Neuron view also shows compactly in the Environment page's drawer", async ({ page }) => {
  await useNeuronDriver(page);
  const compact = page.getByTestId("neuron-view-compact");
  await expect(compact).toBeVisible();
  await expect(compact.getByTestId("neuron-z")).toHaveText("0.00");
  await expect(compact.getByTestId("steer-dial")).toBeVisible();
});

test("a selected car's own inputs appear in the neuron view", async ({ page }) => {
  await useNeuronDriver(page);
  const cars = page.getByTestId("input-cars");
  await cars.fill("3");
  await cars.press("Enter");
  await page.getByTestId("check-random-start").click();
  await expect(page.getByTestId("check-random-start")).toBeChecked();
  await goToNetwork(page);
  await expect(page.getByText("Neuron of car 1")).toBeVisible();
  const first = await page.getByTestId("neuron-input-0").innerText();

  await page.getByRole("link", { name: "Environment" }).click();
  await page.getByTestId("select-car").selectOption("2");
  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByText("Neuron of car 3")).toBeVisible();
  await expect(page.getByTestId("neuron-input-0")).not.toHaveText(first);
});

test("the neuron keeps its weights and settings through Reset and scenario changes", async ({
  page,
}) => {
  await useNeuronDriver(page);
  await goToNetwork(page);
  await setNumber(page, "w_left", -12);
  await setNumber(page, "w_right", 12);
  await setNumber(page, "bias", 0.1);
  await setNumber(page, "throttle", 0.24);
  await page.getByTestId("select-activation").selectOption("sigmoid");
  await expect(page.getByTestId("select-activation")).toHaveValue("sigmoid");

  const expectKept = async () => {
    await page.getByRole("link", { name: "Network" }).click();
    await expect(page.getByTestId("number-w_left")).toHaveValue("-12.00");
    await expect(page.getByTestId("number-w_right")).toHaveValue("12.00");
    await expect(page.getByTestId("number-bias")).toHaveValue("0.10");
    await expect(page.getByTestId("number-throttle")).toHaveValue("0.24");
    await expect(page.getByTestId("select-activation")).toHaveValue("sigmoid");
    await page.getByRole("link", { name: "Environment" }).click();
  };

  // Drive a little, then Reset.
  await page.getByRole("link", { name: "Environment" }).click();
  await page.getByTestId("btn-step").click();
  await page.getByTestId("btn-reset").click();
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await expectKept();

  // More cars, a random start, and a detour through another driver.
  const cars = page.getByTestId("input-cars");
  await cars.fill("3");
  await cars.press("Enter");
  await expect(page.getByTestId("select-car")).toBeVisible();
  await expectKept();
  await page.getByTestId("check-random-start").click();
  await expect(page.getByTestId("check-random-start")).toBeChecked();
  await expectKept();
  await page.getByTestId("select-policy").selectOption("keyboard");
  await expect(page.getByTestId("select-policy")).toHaveValue("keyboard");
  await page.getByTestId("select-policy").selectOption("neuron");
  await expect(page.getByTestId("select-policy")).toHaveValue("neuron");
  await expectKept();
});
