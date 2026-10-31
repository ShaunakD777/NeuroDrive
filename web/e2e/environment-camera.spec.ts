import { expect, type Locator, type Page, test } from "@playwright/test";

// Mouse pan and zoom in the Environment view, full page and compact drawer (task 4.10). The canvas
// reports its camera as data-camera="cx:cy:scale" (metres, metres, pixels per metre) and its mode
// as data-camera-mode (follow, fit or free).

interface Camera {
  cx: number;
  cy: number;
  scale: number;
}

async function cameraOf(canvas: Locator): Promise<Camera> {
  const text = (await canvas.getAttribute("data-camera")) ?? "";
  const [cx, cy, scale] = text.split(":").map(Number);
  return { cx, cy, scale };
}

/** The world point (metres) drawn at canvas pixel (px, py), as the view's own geometry defines it. */
function screenToWorld(cam: Camera, w: number, h: number, px: number, py: number): [number, number] {
  return [cam.cx + (px - w / 2) / cam.scale, cam.cy - (py - h / 2) / cam.scale];
}

/** Wait for the first frame: before it arrives the canvas reports a placeholder camera (0:0:1). */
async function ready(page: Page, canvas: Locator) {
  await expect(page.getByTestId("connection-badge")).toHaveAttribute("data-status", "connected");
  await expect(canvas).toHaveAttribute("data-cars", /\d/);
  await expect.poll(async () => (await cameraOf(canvas)).scale).not.toBe(1);
}

test("full view: the wheel zooms around the cursor, a drag pans, double-click follows again", async ({
  page,
}) => {
  await page.goto("/#/environment");
  const canvas = page.getByTestId("environment-view").getByTestId("environment-canvas");
  await ready(page, canvas);
  await expect(canvas).toHaveAttribute("data-camera-mode", "follow");

  // Whole track, then one wheel notch in at a point left of centre.
  await page.getByTestId("camera-fit").click();
  await expect(canvas).toHaveAttribute("data-camera-mode", "fit");
  const box = (await canvas.boundingBox())!;
  const before = await cameraOf(canvas);
  const px = box.width * 0.3;
  const py = box.height * 0.4;
  const under = screenToWorld(before, box.width, box.height, px, py);
  await page.mouse.move(box.x + px, box.y + py);
  await page.mouse.wheel(0, -100);
  await expect(canvas).toHaveAttribute("data-camera-mode", "free");
  await expect(page.getByTestId("camera-free")).toBeVisible();
  const zoomed = await cameraOf(canvas);
  expect(zoomed.scale / before.scale).toBeCloseTo(1.1618, 2);
  // The same world point is still under the cursor (within one pixel).
  const after = screenToWorld(zoomed, box.width, box.height, px, py);
  expect(Math.abs(after[0] - under[0]) * zoomed.scale).toBeLessThan(1);
  expect(Math.abs(after[1] - under[1]) * zoomed.scale).toBeLessThan(1);

  // A drag of 60 px right and 40 px up moves the centre by that many pixels in metres.
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2 - 20, { steps: 3 });
  await page.mouse.move(box.x + box.width / 2 + 60, box.y + box.height / 2 - 40, { steps: 3 });
  await page.mouse.up();
  const panned = await cameraOf(canvas);
  expect(panned.cx).toBeCloseTo(zoomed.cx - 60 / zoomed.scale, 2);
  expect(panned.cy).toBeCloseTo(zoomed.cy - 40 / zoomed.scale, 2);
  expect(panned.scale).toBeCloseTo(zoomed.scale, 3);

  // The +/− buttons zoom the free view around its centre.
  await page.getByTestId("zoom-in").click();
  await expect.poll(async () => (await cameraOf(canvas)).scale).toBeGreaterThan(panned.scale);
  const buttonZoom = await cameraOf(canvas);
  expect(buttonZoom.scale / panned.scale).toBeCloseTo(1.25, 2);
  expect(buttonZoom.cx).toBeCloseTo(panned.cx, 2);

  // Whole track and double-click bring the view back.
  await page.getByTestId("camera-fit").click();
  await expect(canvas).toHaveAttribute("data-camera-mode", "fit");
  expect((await cameraOf(canvas)).scale).toBeCloseTo(before.scale, 3);
  await canvas.dblclick({ position: { x: 10, y: 10 } });
  await expect(canvas).toHaveAttribute("data-camera-mode", "follow");
});

test("full view: while following, the wheel changes only the zoom level", async ({ page }) => {
  await page.goto("/#/environment");
  const canvas = page.getByTestId("environment-view").getByTestId("environment-canvas");
  await ready(page, canvas);
  const before = await cameraOf(canvas);
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.2);
  await page.mouse.wheel(0, 100); // one notch out
  await expect.poll(async () => (await cameraOf(canvas)).scale).toBeLessThan(before.scale);
  await expect(canvas).toHaveAttribute("data-camera-mode", "follow");
  const after = await cameraOf(canvas);
  expect(after.cx).toBeCloseTo(before.cx, 2); // still centred on the (paused) car
  expect(after.scale / before.scale).toBeCloseTo(0.8607, 2);
});

test("compact drawer: wheel and drag work, and its Follow car button brings the car back", async ({
  page,
}) => {
  await page.goto("/#/network");
  const compact = page.getByTestId("environment-compact");
  const canvas = compact.getByTestId("environment-canvas");
  await ready(page, canvas);
  await expect(compact.getByTestId("compact-follow")).toHaveCount(0);
  const before = await cameraOf(canvas);
  const box = (await canvas.boundingBox())!;

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 50, box.y + box.height / 2, { steps: 4 });
  await page.mouse.up();
  await expect(canvas).toHaveAttribute("data-camera-mode", "free");
  const panned = await cameraOf(canvas);
  expect(panned.cx).toBeCloseTo(before.cx + 50 / before.scale, 2);

  await page.mouse.move(box.x + box.width * 0.25, box.y + box.height * 0.25);
  await page.mouse.wheel(0, 300); // zoom out
  await expect.poll(async () => (await cameraOf(canvas)).scale).toBeLessThan(panned.scale);

  await compact.getByTestId("compact-follow").click();
  await expect(canvas).toHaveAttribute("data-camera-mode", "follow");
  await expect(compact.getByTestId("compact-follow")).toHaveCount(0);
  expect((await cameraOf(canvas)).scale).toBeCloseTo(before.scale, 3);
});

test("a plain click on the car does not move the free view (a click is not a drag)", async ({ page }) => {
  await page.goto("/#/environment");
  const canvas = page.getByTestId("environment-view").getByTestId("environment-canvas");
  await ready(page, canvas);
  await page.getByTestId("camera-fit").click();
  await expect(canvas).toHaveAttribute("data-camera-mode", "fit");
  // The canvas lists each car's screen position. Zoom in around the car (it stays under the
  // cursor), then click it: a click does not count as a drag.
  const box = (await canvas.boundingBox())!;
  const carAt = async () => (((await canvas.getAttribute("data-cars")) ?? "").split(",")[0] ?? "").split(":").map(Number);
  const [fx, fy] = await carAt();
  await page.mouse.move(box.x + fx, box.y + fy);
  await page.mouse.wheel(0, -200);
  await expect(canvas).toHaveAttribute("data-camera-mode", "free");
  const [cx, cy] = await carAt();
  expect(Math.abs(cx - fx) + Math.abs(cy - fy)).toBeLessThanOrEqual(2);
  const camBefore = await canvas.getAttribute("data-camera");
  await canvas.click({ position: { x: cx, y: cy } });
  await expect(canvas).toHaveAttribute("data-camera", camBefore!); // a click does not move the view
  await expect(canvas).toHaveAttribute("data-camera-mode", "free");
});
