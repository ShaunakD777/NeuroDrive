// Records representative data from a running NeuroDrive server into public/recorded/*.json.
//
// The prototype NEVER talks to the Python server at runtime: it only reads the static files this
// script wrote once. The script is kept so the mock data can be regenerated or replaced; it only
// reads from the server (REST GETs, and the same live commands the web app's buttons send).
//
//   1. start the server:   uv run python -m neurodrive serve        (repository root)
//   2. record:             node scripts/capture-mock-data.mjs       (this folder)
//
// Options (environment variables): ND_API (default http://127.0.0.1:8000), ND_RUN (a GA run id
// trained with task 7.8 or later), ND_GENERATION (default 18), ND_START (default 0).

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const API = process.env.ND_API ?? "http://127.0.0.1:8000";
const RUN = process.env.ND_RUN ?? "20261114-140528-ga-lineage78";
const GENERATION = Number(process.env.ND_GENERATION ?? 18);
const START = Number(process.env.ND_START ?? 0);
const SPEED = 8; // simulation steps per frame sent; frames are interpolated by the prototype
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "recorded");
mkdirSync(join(OUT, "cars"), { recursive: true });

const round = (digits) => (v) => (typeof v === "number" ? Number(v.toFixed(digits)) : v);
const deep = (value, f) =>
  Array.isArray(value)
    ? value.map((v) => deep(v, f))
    : value !== null && typeof value === "object"
      ? Object.fromEntries(Object.entries(value).map(([k, v]) => [k, deep(v, f)]))
      : f(value);
const save = (name, data) => {
  const text = JSON.stringify(data);
  writeFileSync(join(OUT, name), text);
  console.log(`  wrote ${name} (${(text.length / 1024).toFixed(0)} KB)`);
};
const get = async (path) => {
  const response = await fetch(`${API}/api/${path}`);
  if (!response.ok) throw new Error(`GET ${path}: ${response.status}`);
  return response.json();
};

/** One connection to /ws/live with a queue of the messages received. */
function connect() {
  const ws = new WebSocket(`${API.replace(/^http/, "ws")}/ws/live`);
  const queue = [];
  let wake = null;
  ws.addEventListener("message", (event) => {
    queue.push(JSON.parse(event.data));
    wake?.();
  });
  const next = async (accept, timeoutMs = 30000) => {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      while (queue.length > 0) {
        const message = queue.shift();
        if (message.type === "error") console.log("  server error:", message.message);
        if (accept(message)) return message;
      }
      if (Date.now() > deadline) throw new Error("timed out waiting for a message");
      await new Promise((resolve) => {
        wake = resolve;
        setTimeout(resolve, 200);
      });
    }
  };
  const send = (command) => ws.send(JSON.stringify(command));
  const opened = new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve);
    ws.addEventListener("error", reject);
  });
  return { ws, next, send, opened };
}

/** Drive the loaded scenario once with `car` selected; returns every frame and the ending. */
async function drive(live, car) {
  live.send({ type: "reset" });
  await live.next((m) => m.type === "frame" && m.step === 0);
  live.send({ type: "select_car", index: car });
  const network = await live.next((m) => m.type === "network" && m.car === car);
  await live.next((m) => m.type === "frame" && m.selected?.index === car);
  live.send({ type: "set_speed", speed: SPEED });
  live.send({ type: "play" });
  const frames = [];
  let end = null;
  while (end === null) {
    const message = await live.next((m) => m.type === "frame" || m.type === "episode_end", 60000);
    if (message.type === "episode_end") end = message;
    else if (message.selected?.index === car) frames.push(message);
  }
  return { network, frames, end };
}

async function main() {
  console.log(`recording ${RUN}, generation ${GENERATION}, start ${START + 1}`);

  // ── The run's charts (plain REST reads) ─────────────────────────────────────────────
  const detail = await get(`runs/${RUN}`);
  const checkpoints = await get(`runs/${RUN}/checkpoints`);
  const previous = checkpoints.units.filter((u) => u < GENERATION).pop() ?? null;
  const run = {
    run_id: RUN,
    config: detail.config,
    meta: detail.meta,
    checkpoints: checkpoints.units,
    generation: GENERATION,
    start: START,
    evolution: deep(await get(`runs/${RUN}/evolution`), round(4)),
    evolutionStart: deep(await get(`runs/${RUN}/evolution?start=${START}`), round(4)),
    starts: deep(await get(`runs/${RUN}/generations/${GENERATION}/starts`), round(4)),
    ancestry: deep(await get(`runs/${RUN}/generations/${GENERATION}/ancestry`), round(4)),
    exam: deep((await get(`runs/${RUN}/exams/${GENERATION}`)).card, round(4)),
    examFinal: deep((await get(`runs/${RUN}/exams/${checkpoints.units.at(-1)}`)).card, round(4)),
    diff:
      previous === null
        ? null
        : deep(await get(`runs/${RUN}/checkpoints/${GENERATION}/diff?from=${previous}`), round(5)),
  };
  const track = deep(await get(`tracks/${detail.config.fitness.track}`), round(2));
  save("track.json", track);

  // ── The race, once per featured car (the drive is deterministic, so the cars repeat) ──
  const live = connect();
  await live.opened;
  live.send({
    type: "load_scenario",
    track: detail.config.fitness.track,
    policy: "population",
    n_cars: 1,
    seed: null,
    checkpoint: null,
    start_shift: null,
    population: { run: RUN, unit: GENERATION, start: START, ghost_unit: GENERATION - 1 },
  });
  const scenario = await live.next((m) => m.type === "scenario" && m.population !== null);
  const info = scenario.population;
  console.log(`  ${info.size} networks + ghost car ${info.ghost_car}; champion row ${info.champion}`);

  const first = await drive(live, info.champion);
  const status = first.end.status;
  const byRank = Array.from({ length: info.size }, (_, i) => i).sort((a, b) => info.ranks[a] - info.ranks[b]);
  const pick = (test) => byRank.find((i) => i !== info.champion && test(i));
  const featured = [
    info.champion,
    pick((i) => status[i] === "finished" && !info.elite.includes(i)), // the best child that finishes
    pick((i) => status[i] === "crashed"),
    byRank.filter((i) => status[i] === "timeout")[Math.floor(byRank.filter((i) => status[i] === "timeout").length / 2)],
    info.ghost_car,
  ].filter((i) => i !== undefined && i !== null);
  console.log("  featured cars:", featured.join(", "), "→", featured.map((i) => status[i]).join(", "));

  const r2 = round(2);
  const r3 = round(3);
  const columns = (frames) => ({
    step: frames.map((f) => f.step),
    time: frames.map((f) => r3(f.time)),
    x: frames.map((f) => f.cars.x.map(r2)),
    y: frames.map((f) => f.cars.y.map(r2)),
    psi: frames.map((f) => f.cars.psi.map(r3)),
    v: frames.map((f) => f.cars.v.map(round(1))),
    progress: frames.map((f) => f.cars.progress.map(round(1))),
    steer: frames.map((f) => f.cars.action.map((a) => r2(a[0]))),
    throttle: frames.map((f) => f.cars.action.map((a) => r2(a[1]))),
    status: frames.map((f) => f.cars.status),
  });
  save("race.json", {
    scenario: deep(scenario, round(4)),
    stepsPerFrame: SPEED,
    frames: columns(first.frames),
    end: deep(first.end, round(4)),
    featured,
  });

  for (const car of featured) {
    const recorded = car === info.champion ? first : await drive(live, car);
    const mutation =
      car < info.size && GENERATION > 1
        ? await get(`runs/${RUN}/generations/${GENERATION}/rows/${car}/mutation`).catch(() => null)
        : null;
    save(`cars/${car}.json`, {
      car,
      network: deep(recorded.network, round(4)),
      mutation: mutation === null ? null : deep(mutation, round(5)),
      frames: recorded.frames.map((f) => ({
        step: f.step,
        action: f.cars.action[car].map(r3),
        rays: f.selected.rays.map(r2),
        observation: f.selected.observation.map(round(4)),
        trace: deep(f.selected.trace, r3),
        explain: deep(f.selected.explain, round(4)),
      })),
    });
  }
  live.ws.close();
  save("run.json", run);

  // ── Compare runs: optimizer sweeps (loss) and imitation sweeps (validation loss) ───
  const runs = (await get("runs")).runs;
  const sweepIds = (name) => runs.filter((r) => (r.sweep ?? "").endsWith(name)).map((r) => r.run_id);
  const compare = async (names, metric) => {
    const ids = names.flatMap(sweepIds);
    if (ids.length === 0) return null;
    const answer = await get(`runs/compare?ids=${ids.join(",")}&metric=${metric}`);
    // Keep about 150 points per curve (every early unit, then evenly spaced on a log axis): the
    // charts are a design reference, and 3,000 points per run would make the file 2 MB.
    const keep = (n) => {
      const picked = new Set([n - 1]);
      for (let i = 0; i < Math.min(n, 30); i++) picked.add(i);
      for (let k = 0; k <= 120; k++) picked.add(Math.min(n - 1, Math.round(30 * (n / 30) ** (k / 120))));
      return [...picked].sort((a, b) => a - b);
    };
    const thin = (item) => {
      const idx = keep(item.unit.length);
      const out = { ...item };
      for (const key of ["unit", "median", "q1", "q3", "values", "n_present", "n_finite"]) {
        if (Array.isArray(item[key])) out[key] = idx.map((i) => item[key][i]);
      }
      return out;
    };
    const sig = (v) => (typeof v === "number" ? Number(v.toPrecision(4)) : v);
    return deep({ ...answer, groups: answer.groups.map(thin), runs: answer.runs.map(thin) }, sig);
  };
  save("compare.json", {
    optimizers: await compare(["toy_sine-s3-sgd", "toy_sine-s3-momentum", "toy_sine-s3-rmsprop", "toy_sine-s3-adam"], "loss"),
    imitation: await compare(["imitation-s5c-efull", "imitation-s5c-human", "dagger-s5c-dw"], "val_loss"),
  });

  // ── The Driving Exam of the last checkpoint of every seed, per method (the common yardstick) ──
  const lastCard = async (r) => {
    if (r.last_unit === null) return null;
    return get(`runs/${r.run_id}/exams/${r.last_unit}`).then((x) => x.card).catch(() => null);
  };
  const groups = [
    { key: "expert", label: "Scripted expert", runs: runs.filter((r) => r.method === "scripted_expert").slice(0, 1) },
    { key: "neuron", label: "Hand-wired neuron", runs: runs.filter((r) => r.method === "hand_neuron").slice(0, 1) },
    { key: "random", label: "Random networks", runs: runs.filter((r) => (r.sweep ?? "") === "20261111-073907-random_network-baseline") },
    { key: "random_search", label: "Random search", runs: runs.filter((r) => (r.sweep ?? "").endsWith("random_search-s4")) },
    { key: "imitation", label: "Imitation (expert)", runs: runs.filter((r) => (r.sweep ?? "").endsWith("imitation-s5c-efull")) },
    { key: "dagger", label: "DAgger", runs: runs.filter((r) => (r.sweep ?? "").endsWith("dagger-s5c-dw")) },
    { key: "ga", label: "Genetic algorithm", runs: runs.filter((r) => r.run_id === RUN) },
  ];
  const exams = [];
  for (const g of groups) {
    const cards = [];
    for (const r of g.runs) {
      const card =
        g.key === "expert" || g.key === "neuron" || g.key === "random"
          ? await get(`runs/${r.run_id}/exams/0`).then((x) => x.card).catch(() => null)
          : await lastCard(r);
      if (card !== null) cards.push({ run: r.run_id, training: card.training, heldout: card.heldout });
    }
    exams.push({ key: g.key, label: g.label, seeds: cards.length, cards: deep(cards, round(4)) });
    console.log(`  exam cards: ${g.label} × ${cards.length}`);
  }
  save("exams.json", exams);
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error);
    process.exit(1);
  },
);
