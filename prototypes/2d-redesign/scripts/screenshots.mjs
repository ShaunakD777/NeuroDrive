// Takes the screenshots in screenshots/ from the running prototype.
//
//   1. npm run dev            (this folder; serves http://127.0.0.1:5180)
//   2. npm run screenshots
//
// Every image is the real prototype in a real browser (Chromium, through Playwright). The script
// drives it the way a user would where it can (clicks, hovers), and sets the playback position
// and theme through the store (window.__nd) so the images are the same on every run.

import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const BASE = process.env.ND_PROTOTYPE ?? "http://127.0.0.1:5180";
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "screenshots");
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const errors = [];
let page;

async function open(width = 1440, height = 900) {
  await page?.close();
  page = await browser.newPage({ viewport: { width, height } });
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${BASE}/#/environment`).catch(() => {
    console.error(`Cannot reach ${BASE}. Start the prototype first: npm run dev`);
    process.exit(1);
  });
  await page.waitForFunction(() => window.__nd?.getState().data !== null);
}

const set = (state) => page.evaluate((s) => window.__nd.setState(s), state);
const go = async (view) => {
  await page.evaluate((v) => (window.location.hash = `#/${v}`), view);
  await page.waitForTimeout(250);
};
const click = (testId) => page.locator(`[data-testid="${testId}"]`).first().click();
const settle = (ms = 700) => page.waitForTimeout(ms);
const shot = async (name) => {
  await page.screenshot({ path: join(OUT, `${name}.png`) });
  console.log(`  ${name}.png`);
};
/** A clean slate between shots: paused at the default moment, champion selected, nothing picked. */
const reset = async (theme, extra = {}) => {
  await set({
    theme,
    playing: false,
    t: 118,
    viewKey: null,
    whatIf: {},
    neuron: null,
    connection: null,
    output: null,
    peek: null,
    car: 46,
    cameraMode: "follow",
    lineMode: "weights",
    layout: "page",
    drawerOpen: true,
    drawerTab: "network",
    notes: false,
    experimentTab: "timeline",
    generation: 18,
    evoStart: null,
    showRays: true,
    showTrail: true,
    ...extra,
  });
};

// ── Environment ──────────────────────────────────────────────────────────────────────────
await open();
for (const theme of ["light", "dark"]) {
  await reset(theme);
  await go("environment");
  await settle(1200);
  await shot(`environment-follow-${theme}`);

  await reset(theme, { cameraMode: "fit", t: 262, drawerTab: "standings" });
  await settle(1200);
  await shot(`environment-whole-track-standings-${theme}`);
}

// An output selected: rays coloured by sensitivity, trail coloured by the decision.
await reset("light", { output: "steer", neuron: { layer: 3, index: 0 } });
await settle(900);
await shot("environment-steer-selected-light");

// Looking back at an earlier step of the drive.
await reset("dark", { viewKey: 92 });
await settle(1200);
await shot("environment-look-back-dark");

// A crashed car, selected.
await reset("light", { car: 30, t: 158 });
await settle(1200);
await shot("environment-crashed-car-light");

// The ghost selected (last generation's champion).
await reset("dark", { car: 50, t: 150 });
await settle(1200);
await shot("environment-ghost-selected-dark");

// ── Network ──────────────────────────────────────────────────────────────────────────────
for (const theme of ["light", "dark"]) {
  await reset(theme);
  await go("network");
  await settle(900);
  await shot(`network-weights-${theme}`);
  await click("inspect-steer");
  await settle(700);
  await shot(`network-steer-explained-${theme}`);
}

await reset("dark", { lineMode: "flow" });
await settle(700);
await shot("network-flow-dark");

await reset("light", { lineMode: "mutation" });
await settle(700);
await shot("network-mutation-light");

// Hover: the floating read-out on a hidden neuron, related connections lit.
await reset("light");
await settle(500);
await page.locator('[data-testid="net-node-2-4"]').hover();
await settle(600);
await shot("network-hover-neuron-light");

// A hidden neuron kept selected.
await reset("dark", { neuron: { layer: 1, index: 6 } });
await page.mouse.move(5, 300);
await settle(600);
await shot("network-neuron-selected-dark");

// What-if: move the centre ray's slider; the ghost car appears on the track (drawer).
await reset("light", { output: "steer", neuron: { layer: 3, index: 0 }, whatIf: { 2: 0.15, 1: 0.08 } });
await settle(600);
await page.locator('[data-testid="whatif-panel"]').scrollIntoViewIfNeeded();
await settle(600);
await shot("network-what-if-light");
await go("environment");
await set({ drawerTab: "network" });
await settle(1200);
await shot("environment-what-if-ghost-light");

// ── Experiment ───────────────────────────────────────────────────────────────────────────
await open(1440, 1760);
for (const theme of ["light", "dark"]) {
  await reset(theme);
  await go("experiment");
  await settle(1000);
  await shot(`experiment-timeline-${theme}`);
}
await open(1440, 1160);
for (const theme of ["light", "dark"]) {
  await reset(theme, { experimentTab: "compare" });
  await go("experiment");
  await settle(1000);
  await shot(`experiment-compare-${theme}`);
}
await open();
await reset("light", { experimentTab: "training" });
await go("experiment");
await settle(1500);
await shot("experiment-training-light");

// Chart hover: the read-out on the fitness chart.
await reset("dark");
await settle(800);
const chart = page.locator('[data-testid="evolution-fitness"]');
const box = await chart.boundingBox();
await page.mouse.move(box.x + box.width * 0.72, box.y + box.height * 0.4);
await settle(500);
await shot("experiment-chart-hover-dark");

// ── Layouts ──────────────────────────────────────────────────────────────────────────────
await open(1680, 940);
await reset("dark", { layout: "split", output: "throttle", neuron: { layer: 3, index: 1 } });
await go("environment");
await settle(1300);
await shot("layout-split-environment-network-dark");

await open(390, 844);
for (const view of ["environment", "network", "experiment"]) {
  await reset("light", { drawerOpen: false });
  await go(view);
  await settle(1200);
  await shot(`phone-${view}-light`);
}

// ── Annotated mockups: the design notes layer ────────────────────────────────────────────
await open(1600, 1000);
await reset("light", { notes: true });
await go("environment");
await settle(1300);
await shot("annotated-environment");
await reset("light", { notes: true, output: "steer", neuron: { layer: 3, index: 0 } });
await go("network");
await settle(1000);
await shot("annotated-network");
await open(1600, 1800);
await reset("light", { notes: true });
await go("experiment");
await settle(1200);
await shot("annotated-experiment-timeline");
await open(1600, 1160);
await reset("light", { notes: true, experimentTab: "compare" });
await go("experiment");
await settle(1200);
await shot("annotated-experiment-compare");

await browser.close();
if (errors.length > 0) {
  console.error(`\n${errors.length} console error(s):\n${[...new Set(errors)].slice(0, 10).join("\n")}`);
  process.exit(1);
}
console.log("\nno console errors");
