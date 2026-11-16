// The recorded adapter: the world snapshot built from public/recorded/ and the prototype's clock.
// Everything that stands in for Python is in src/mock/mockServer.ts; this file only picks the
// right recorded numbers for the moment shown.

import { castRay, contributionsOf, detailAt, poseAt, rollout, whatIf } from "../mock/mockServer";
import type { State } from "../state/store";
import type { CarDetail, Mutation, Network, Scenario, Track, TrailView, WhatIfView, World } from "./model";

type Inputs = Pick<State, "data" | "t" | "car" | "viewKey" | "playing" | "speed" | "whatIf">;

/** The recorded scenario in the shape of the server's `scenario` message. */
function scenarioOf(data: NonNullable<State["data"]>): Scenario {
  return { ...data.race.scenario, policy: "population", checkpoint: null, seed: null, start_shift: null } as Scenario;
}

export function buildRecorded(s: Inputs): World | null {
  const data = s.data;
  if (data === null) return null;
  const { race } = data;
  const track = data.track as Track;
  const scenario = scenarioOf(data);
  const sc = race.scenario;
  const frames = race.frames;
  const last = frames.step.length - 1;
  const t = Math.max(0, Math.min(last, s.t));
  const key = Math.floor(t);
  const step = Math.round(frames.step[key] + (t - key) * race.stepsPerFrame);

  const cars = Array.from({ length: sc.n_cars }, (_, i) => poseAt(race, t, i));
  const looking = s.viewKey !== null;
  const shownT = s.viewKey ?? t;
  const record = data.cars.get(s.car) ?? null;
  const pose = looking ? poseAt(race, shownT, s.car) : cars[s.car];
  const detail = record === null ? null : detailAt(record, shownT);
  const shown: CarDetail = {
    index: s.car,
    step: looking ? frames.step[s.viewKey ?? 0] : step,
    pose,
    // Only five cars were recorded with their sensors; for the others the stand-in measures the
    // rays from the track edges (production: every selected car's rays come from Python).
    rays: detail?.rays ?? sc.ray_angles.map((a) => castRay(data.track, pose.x, pose.y, pose.psi + a, sc.ray_range)),
    observation: detail?.observation ?? [],
    action: [pose.steer, pose.throttle],
    trace: detail?.trace ?? null,
    explain: detail?.explain ?? null,
    neuron: null,
  };

  // The trail: the recorded frames of the selected car over the remembered 60 s.
  const back = Math.round(sc.trail_seconds / (race.stepsPerFrame * sc.dt));
  const trail: TrailView = { steps: [], x: [], y: [], action: [] };
  for (let k = Math.max(0, key - back); k <= key; k++) {
    trail.steps.push(frames.step[k]);
    trail.x.push(frames.x[k][s.car]);
    trail.y.push(frames.y[k][s.car]);
    trail.action.push([frames.steer[k][s.car], frames.throttle[k][s.car]]);
  }

  // What-if: the stand-in runs the recorded network and rolls the actions forward for 2 s.
  let what: WhatIfView | null = null;
  if (!s.playing && !looking && record !== null && detail !== null && Object.keys(s.whatIf).length > 0) {
    const answer = whatIf(record, detail, s.whatIf);
    const real = rollout(cars[s.car], answer.base, 2, sc.dt);
    const tried = rollout(cars[s.car], answer.tried, 2, sc.dt);
    what = { ...answer, ghost: { real: real.path, whatIf: tried.path, end: tried.end } };
  }

  const info = sc.population;
  const isGhost = s.car === info.ghost_car;
  return {
    source: "recorded",
    track,
    scenario,
    step,
    time: step * sc.dt,
    playing: s.playing,
    speed: s.speed,
    ended: t >= last,
    cars,
    selected: s.car,
    lookStep: looking ? shown.step : null,
    shown,
    trail,
    whatIf: what,
    network: (record?.network as Network | undefined) ?? null,
    origin: {
      mutation: (record?.mutation as Mutation | null | undefined) ?? null,
      diff: null,
      diffFrom: null,
      note: isGhost ? `the champion of generation ${info.ghost_unit}, racing as a ghost` : null,
    },
    flow: record !== null && detail !== null ? contributionsOf(record.network, detail.trace.activations) : null,
    endFitness: t >= last ? race.end.fitness : null,
  };
}

/** The cars the recording has networks for (live: every car). */
export function featuredCars(data: State["data"]): number[] | null {
  return data === null ? null : data.race.featured;
}
