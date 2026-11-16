// Drives the prototype against the real Python server and checks that every driver, the
// selection, look-back and what-if work on live data. Saves screenshots to screenshots/live/.
//
//   1. uv run python -m neurodrive serve      (repository root; port 8000)
//   2. npm run dev                            (this folder; port 5180)
//   3. npm run live-check
//
// It reads the world snapshot the views draw (window.__ndWorld) and production's own stores
// (window.__ndProd) to wait for the server's answers instead of guessing with timeouts. Exits
// with 1 on any failed check or any console error. It only uses the server the way the app does:
// it loads scenarios, plays, pauses and asks what-if questions; it trains nothing.

import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const BASE = process.env.ND_PROTOTYPE ?? "http://127.0.0.1:5180";
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "screenshots", "live");
const SHOTS = !process.argv.includes("--no-shots");
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
const failures = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));
// A failed request is logged by the browser as a console error without its address: keep the address.
page.on("response", (r) => r.status() >= 400 && errors.push(`${r.status()} ${new URL(r.url()).pathname}${new URL(r.url()).search}`));

const world = () => page.evaluate(() => window.__ndWorld());
const live = (fn) => page.evaluate(fn);
const settle = (ms = 600) => page.waitForTimeout(ms);
const go = async (view) => {
  await page.evaluate((v) => (window.location.hash = `#/${v}`), view);
  await settle(300);
};
const shot = async (name) => {
  if (!SHOTS) return;
  await page.mouse.move(2, 2); // park the pointer, so no hover tooltip is in the picture
  await settle(150);
  await page.screenshot({ path: join(OUT, `${name}.png`) });
  console.log(`    ${name}.png`);
};
function check(ok, what) {
  console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`);
  if (!ok) failures.push(what);
}
/** Wait until `predicate(arg)` holds in the page (it is evaluated there). */
async function until(predicate, what, timeout = 8000, arg = null) {
  try {
    await page.waitForFunction(predicate, arg, { timeout, polling: 100 });
    return true;
  } catch {
    check(false, `timed out waiting for ${what}`);
    return false;
  }
}
/** Load a scenario the way the scenario bar does (production's loadScenario), and wait for it. */
async function load(changes, what) {
  await page.evaluate((c) => window.__ndProd.useLive.getState().loadScenario(c), changes);
  await until(
    (c) => {
      const w = window.__ndWorld();
      return w !== null && w.scenario.policy === c.policy && w.step === 0 && (c.n_cars === undefined || w.cars.length === c.n_cars);
    },
    what,
    8000,
    changes,
  );
}
async function playFor(ms) {
  await page.locator('[data-testid="btn-play"]').click();
  await settle(ms);
  if ((await world()).playing) await page.locator('[data-testid="btn-play"]').click(); // pause
  await until(() => window.__ndWorld()?.playing === false, "the pause");
}

await page.goto(`${BASE}/?source=live#/environment`).catch(() => {
  console.error(`Cannot reach ${BASE}. Start the prototype first: npm run dev`);
  process.exit(1);
});
await page.evaluate(() => window.__nd.setState({ theme: "light" }));
console.log("Connection");
const connected = await until(() => window.__nd.getState().liveStatus === "connected" && window.__ndWorld() !== null, "the server and its first frame", 15000);
if (!connected) {
  console.error("No live world: is `uv run python -m neurodrive serve` running on port 8000?");
  await browser.close();
  process.exit(1);
}
check((await page.locator('[data-testid="source-switch"]').getAttribute("data-status")) === "connected", "the top bar says Live");

// ── Drivers without a network ────────────────────────────────────────────────────────────
console.log("Keyboard driver");
await load({ policy: "keyboard", n_cars: 1, seed: null }, "the keyboard scenario");
await page.locator('[data-testid="btn-play"]').click();
await page.keyboard.down("ArrowUp");
await settle(1500);
await page.keyboard.up("ArrowUp");
await page.locator('[data-testid="btn-play"]').click();
await until(() => window.__ndWorld()?.playing === false, "the pause");
let w = await world();
check(w.cars[0].v > 1, `the arrow key accelerated the car (v = ${w.cars[0].v.toFixed(2)} m/s)`);
check(w.shown !== null && w.shown.rays.length === 5, "five rays for the selected car");
check((await page.locator('[data-testid="btn-record"]').count()) === 1, "the record bar is shown for the keyboard driver");
await shot("live-environment-keyboard-light");
await go("network");
check((await page.locator('[data-testid="network-empty"]').count()) === 1, "Network view: the keyboard has no network, and says so");
await go("environment");

console.log("Expert, 1 car, dark");
await page.evaluate(() => window.__nd.setState({ theme: "dark" }));
await load({ policy: "expert", n_cars: 1 }, "the expert");
await playFor(2500);
w = await world();
check(w.cars[0].progress > 10, `the expert drives (progress ${w.cars[0].progress.toFixed(1)} m)`);
await shot("live-environment-expert-dark");

console.log("Neuron driver");
await load({ policy: "neuron", n_cars: 1 }, "the neuron driver");
await playFor(800);
await go("network");
check((await page.locator('[data-testid="neuron-panel"]').count()) === 1, "Network view shows the neuron panel");
const before = await live(() => window.__ndWorld().shown.neuron.weights[0]);
await page.locator('[data-testid="param-w_left"]').fill("5");
await until(() => window.__ndWorld()?.shown?.neuron?.weights[0] === 5, "the server to apply w_left = 5");
check(true, `set_param round trip (w_left ${before} → 5)`);
await shot("live-network-neuron-dark");
await go("environment");

// ── Random networks: many cars, liveries, selection, the Network view ────────────────────
console.log("Random networks, 8 cars (liveries) and 100 cars");
await page.evaluate(() => window.__nd.setState({ theme: "light", cameraMode: "fit" }));
await load({ policy: "random_networks", n_cars: 8, seed: null }, "8 random networks");
await playFor(1200);
await shot("live-environment-random8-whole-track-light");
await load({ policy: "random_networks", n_cars: 100, seed: 3 }, "100 random networks");
await playFor(1000);
w = await world();
check(w.cars.length === 100, "100 cars in the frame");
await page.evaluate(() => window.__nd.getState().selectCar(42));
await until(() => window.__ndWorld()?.selected === 42 && window.__ndWorld()?.network?.car === 42, "car 43 and its network");
check(true, "selecting car 43 brought its network");
await shot("live-environment-random100-whole-track-light");
await page.evaluate(() => window.__nd.setState({ cameraMode: "follow" }));
await go("network");
w = await world();
check(w.network !== null && w.shown?.trace !== null && w.shown?.explain !== null, "network, trace and explanation of car 43");
check((await page.locator('[data-testid^="net-edge-"]').count()) > 50, "the diagram drew its connections");
await page.locator('[data-testid="inspect-steer"]').click();
check((await page.locator('[data-testid="inspector-why"]').count()) === 1, "the inspector explains Steer");
check(await page.locator("button", { hasText: "Flow" }).isDisabled(), "Flow is disabled live (needs F1)");
await shot("live-network-random-steer-light");

console.log("What-if");
await page.locator('[data-testid="whatif-2"]').fill("0.05");
await until(() => window.__ndWorld()?.whatIf?.ghost != null, "Python's what-if answer with its ghost");
w = await world();
check(w.whatIf.ghost.whatIf.length > 10, `the ghost path from Python (${w.whatIf.ghost.whatIf.length} points)`);
await shot("live-network-what-if-light");
await go("environment");
await settle(400);
check((await page.locator('[data-testid="ghost-legend"]').count()) === 1, "the track shows the what-if legend");
await shot("live-environment-what-if-ghost-light");
await page.evaluate(() => window.__nd.getState().clearWhatIf());

console.log("Look back");
await playFor(1500);
w = await world();
const target = w.trail.steps[Math.floor(w.trail.steps.length / 3)];
await page.evaluate((s) => window.__nd.getState().lookAt(s), target);
await until((s) => window.__ndWorld()?.lookStep === s, `the moment at step ${target}`, 8000, target);
check((await page.locator('[data-testid="back-to-now"]').count()) === 1, "the look-back strip offers Back to now");
await shot("live-environment-look-back-light");
await page.locator('[data-testid="back-to-now"]').click();
await until(() => window.__ndWorld()?.lookStep === null, "back to now");

// ── A GA generation ─────────────────────────────────────────────────────────────────────
console.log("GA generation");
const gaRun = await page.evaluate(async () => {
  const runs = (await (await fetch("/api/runs")).json()).runs.filter((r) => r.method === "ga" && r.checkpoints > 0);
  return runs[0] ?? null;
});
if (gaRun === null) {
  check(true, "no GA run in runs/: skipped (train one with configs/ga.yaml)");
} else {
  const unit = Math.max(2, Math.min(gaRun.last_unit ?? 2, 12));
  await page.evaluate(
    ([run, u]) =>
      window.__ndProd.useLive.getState().loadScenario({ policy: "population", population: { run, unit: u, start: 0, ghost_unit: u - 1 } }),
    [gaRun.run_id, unit],
  );
  await until((u) => window.__ndWorld()?.scenario.population?.unit === u && window.__ndWorld()?.step === 0, `generation ${unit}`, 20000, unit);
  w = await world();
  check(w.scenario.population.size + 1 === w.cars.length, `${w.scenario.population.size} networks and a ghost`);
  await page.evaluate(() => window.__nd.setState({ cameraMode: "fit", drawerTab: "standings", theme: "dark" }));
  await playFor(2500);
  await shot("live-environment-ga-standings-dark");
  const child = await page.evaluate(() => {
    const p = window.__ndWorld().scenario.population;
    return p.ranks.map((r, i) => i).find((i) => !p.elite.includes(i)) ?? 0;
  });
  await page.locator('[data-testid="standings-row"]').nth(3).click();
  await page.evaluate((i) => window.__nd.getState().selectCar(i), child);
  await until(() => window.__ndWorld()?.origin.mutation !== null || window.__ndWorld()?.origin.note !== null, "the lineage answer");
  await go("network");
  check((await page.locator('[data-testid="lineage-banner"]').count()) === 1, "the lineage banner");
  if (!(await page.locator("button", { hasText: "Mutation" }).isDisabled())) {
    await page.locator("button", { hasText: "Mutation" }).click();
    await settle(300);
    await shot("live-network-ga-mutation-dark");
  }
  await page.evaluate(() => window.__nd.setState({ cameraMode: "follow", drawerTab: "network" }));
  await go("environment");
}

// ── The Experiment view on the server's run folders ─────────────────────────────────────
console.log("Experiment: Timeline of a GA run");
const runList = await page.evaluate(async () => (await (await fetch("/api/runs")).json()).runs);
const ga = runList.find((r) => r.method === "ga" && r.checkpoints > 0);
const other = runList.find((r) => !r.method.startsWith("toy_") && r.method !== "ga" && r.checkpoints > 0);
await page.evaluate(() => window.__nd.setState({ theme: "light", experimentTab: "timeline", drawerOpen: true }));
await go("experiment");
if (ga === undefined) {
  check(true, "no GA run in runs/: Timeline GA checks skipped");
} else {
  await page.evaluate((id) => window.__nd.setState({ runId: id }), ga.run_id);
  const drawn = await until(() => document.querySelector('[data-testid="evolution-fitness"] path.line') !== null, "the GA run's fitness chart", 20000);
  if (drawn) check(true, `fitness chart of ${ga.run_id.replace(/^\d{8}-\d{6}-/, "")}`);
  await page.evaluate(() => window.__nd.setState({ generation: 5 }));
  await until(
    () => document.querySelector('[data-testid="evolution-outcome-cell"]') !== null || document.querySelector('[data-testid="outcomes-figure"] [data-testid="panel-missing"]') !== null,
    "generation 5's outcome grid (or its reason)",
    20000,
  );
  await until(
    () => document.querySelector('[data-testid="ancestry-chart"]') !== null || document.querySelector('[data-testid="ancestry-figure"] [data-testid="panel-missing"]') !== null,
    "generation 5's ancestry (or its reason)",
    20000,
  );
  await until(
    () => document.querySelector('[data-testid="exam-card"]') !== null || document.querySelector('[data-testid="exam-figure"] [data-testid="panel-missing"]') !== null,
    "the exam panel (a card or its reason)",
    20000,
  );
  check((await page.locator('[data-testid="dash-gen-value"], .dash-gen-value').first().textContent()) === "5", "the generation cursor reads 5");
  await shot("live-experiment-timeline-ga-light");
  if ((await page.locator('[data-testid="evolution-outcome-cell"]').count()) > 0) {
    await page.locator('[data-testid="evolution-outcome-cell"]').nth(4).click();
    await until(() => window.__ndWorld()?.scenario.population?.unit === 5, "the click to race generation 5", 20000);
    check(true, "an outcome cell raced its generation");
  }
  await page.evaluate(() => window.__nd.setState({ theme: "dark" }));
  await settle(300);
  await shot("live-experiment-timeline-ga-dark");
}

if (other !== undefined) {
  console.log(`Experiment: Timeline of a ${other.method} run`);
  await page.evaluate((id) => window.__nd.setState({ runId: id, theme: "light" }), other.run_id);
  const ok = await until(() => document.querySelector('[data-testid="metric-chart"] path.line') !== null, "its metric chart", 20000);
  if (ok) check(true, `metric chart of ${other.run_id.replace(/^\d{8}-\d{6}-/, "")}`);
  await shot("live-experiment-timeline-other-light");
}

console.log("Experiment: Compare runs");
await page.evaluate(() => window.__nd.setState({ experimentTab: "compare", theme: "light" }));
const compared = await until(
  () =>
    document.querySelector('[data-testid="compare-median"]') !== null ||
    (document.querySelector('[data-testid="runlist"]') !== null && document.querySelectorAll('[data-testid="runlist"] input').length === 0),
  "the newest sweep's comparison",
  30000,
);
if (compared) check((await page.locator('[data-testid="compare-median"]').count()) > 0 || (await page.locator('[data-testid="runlist"] input').count()) === 0, "median curves from GET runs/compare (or no run with a loss)");
await shot("live-experiment-compare-light");

console.log("Experiment: Training");
await page.evaluate(() => window.__nd.setState({ experimentTab: "training" }));
// /ws/runs reports runs that are running or were launched by this server, not every old folder.
const listed = await until(() => document.querySelector('[data-testid="training-runs"]')?.previousElementSibling?.textContent === "Runs", "the /ws/runs socket", 20000);
if (listed) check(true, `the run list from /ws/runs (${await page.locator('[data-testid="training-runs"] .runlist-row').count()} rows)`);
await until(() => document.querySelector('[data-testid="live-chart"]') !== null || document.querySelector('[data-testid="live-waiting"]') !== null, "the watched run's chart");
await shot("live-experiment-training-light");
await page.evaluate(() => window.__nd.setState({ experimentTab: "timeline" }));

// ── Production untouched: the live store is production's ───────────────────────────────
const sameStore = await page.evaluate(() => typeof window.__ndProd.useLive.getState().loadScenario === "function");
check(sameStore, "the live source runs on production's useLive store");

console.log("");
if (errors.length > 0) {
  console.log(`Console errors (${errors.length}):`);
  for (const e of [...new Set(errors)].slice(0, 20)) console.log(`  ${e}`);
}
console.log(failures.length === 0 && errors.length === 0 ? "All live checks passed." : `${failures.length} failed checks, ${errors.length} console errors.`);
await browser.close();
process.exit(failures.length === 0 && errors.length === 0 ? 0 : 1);
