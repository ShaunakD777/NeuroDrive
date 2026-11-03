import { expect, test } from "@playwright/test";

// The scripted expert as a driver in the app (task 5.3 follow-up, api v1.12, ADR-035). The
// expert itself lives in Python (env/experts.py); the page only picks it and draws the drive.

test("the expert drives through both hairpins and the Network page explains it has no network", async ({
  page,
}) => {
  await page.goto("/#/environment");
  await expect(page.getByTestId("hud-step")).toHaveText("0");
  await page.getByTestId("select-track").selectOption("hairpins_01");
  await expect(page.getByTestId("select-track")).toHaveValue("hairpins_01");
  await page.getByTestId("select-policy").selectOption("expert");
  await expect(page.getByTestId("select-policy")).toHaveValue("expert");
  await expect(page.getByTestId("hud-step")).toHaveText("0");

  // The hairpins are about 300 m and 500 m in; 600 m means the car went through both.
  await page.getByTestId("select-speed").selectOption("8");
  await page.getByTestId("btn-play").click();
  await expect
    .poll(async () => Number((await page.getByTestId("hud-progress").innerText()).split(" ")[0]), {
      timeout: 30_000,
    })
    .toBeGreaterThan(600);
  await expect(page.getByTestId("hud-status")).toHaveText("Running");

  await page.getByRole("link", { name: "Network" }).click();
  await expect(page.getByTestId("expert-note")).toContainText("The expert has no network");
});
