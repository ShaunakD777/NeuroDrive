// A STAND-IN FOR THE PYTHON SERVER. Everything in this file exists only so the prototype can be
// clicked through without `nd serve`. In production none of it runs in the browser: frames,
// traces, explanations, what-if answers and ghost paths all come from Python (CLAUDE.md §4,
// "Python is the single source of truth"). Do not copy this file into web/.
//
// What it does:
//   - loads the recorded files (one real GA generation, recorded by scripts/capture-mock-data.mjs);
//   - returns the frame at any playback time by interpolating between recorded frames;
//   - answers a what-if question by running the recorded network on changed inputs;
//   - rolls a held action forward for 2 s to draw the what-if ghost (physics v1 equations);
//   - measures ray lengths for cars whose sensor readings were not recorded;
//   - computes weight × value for every connection (the proposed "Flow" mode);
//   - samples the activation functions for the activation plot.

import type { CarFrame, CarRecord, Network, Race, Recorded, Track } from "./types";

const BASE = `${import.meta.env.BASE_URL}recorded/`;

async function getJson<T>(name: string): Promise<T> {
  const response = await fetch(BASE + name);
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return (await response.json()) as T;
}

export async function loadRecorded(): Promise<Recorded> {
  const [track, race, run, compare, exams] = await Promise.all([
    getJson<Recorded["track"]>("track.json"),
    getJson<Race>("race.json"),
    getJson<Recorded["run"]>("run.json"),
    getJson<Recorded["compare"]>("compare.json"),
    getJson<Recorded["exams"]>("exams.json"),
  ]);
  const records = await Promise.all(race.featured.map((car) => getJson<CarRecord>(`cars/${car}.json`)));
  return { track, race, run, compare, exams, cars: new Map(records.map((r) => [r.car, r])) };
}

// ── Frames ────────────────────────────────────────────────────────────────────────────────

export interface CarPose {
  x: number;
  y: number;
  psi: number;
  v: number;
  progress: number;
  steer: number;
  throttle: number;
  status: number;
}

const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

/** Interpolate an angle the short way round. */
function lerpAngle(a: number, b: number, u: number): number {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * u;
}

/** Car `i` at playback position `t` (a fractional index into the recorded frames). */
export function poseAt(race: Race, t: number, i: number): CarPose {
  const f = race.frames;
  const last = f.step.length - 1;
  const a = Math.max(0, Math.min(last, Math.floor(t)));
  const b = Math.min(last, a + 1);
  const u = Math.max(0, Math.min(1, t - a));
  return {
    x: lerp(f.x[a][i], f.x[b][i], u),
    y: lerp(f.y[a][i], f.y[b][i], u),
    psi: lerpAngle(f.psi[a][i], f.psi[b][i], u),
    v: lerp(f.v[a][i], f.v[b][i], u),
    progress: lerp(f.progress[a][i], f.progress[b][i], u),
    steer: lerp(f.steer[a][i], f.steer[b][i], u),
    throttle: lerp(f.throttle[a][i], f.throttle[b][i], u),
    status: f.status[a][i],
  };
}

/** The recorded detail of a featured car at `t`: numbers interpolated, so playing looks smooth. */
export function detailAt(record: CarRecord, t: number): CarFrame {
  const last = record.frames.length - 1;
  const a = Math.max(0, Math.min(last, Math.floor(t)));
  const b = Math.min(last, a + 1);
  const u = Math.max(0, Math.min(1, t - a));
  const A = record.frames[a];
  if (u === 0 || a === b) return A;
  const B = record.frames[b];
  const mix = (p: number[], q: number[]) => p.map((v, k) => lerp(v, q[k] ?? v, u));
  return {
    step: Math.round(lerp(A.step, B.step, u)),
    action: mix(A.action, B.action),
    rays: mix(A.rays, B.rays),
    observation: mix(A.observation, B.observation),
    trace: {
      activations: A.trace.activations.map((row, l) => mix(row, B.trace.activations[l])),
      weighted_sums: A.trace.weighted_sums.map((row, l) => mix(row, B.trace.weighted_sums[l])),
    },
    explain: {
      receipts: A.explain.receipts.map((r, m) => {
        const q = B.explain.receipts[m];
        return {
          output: r.output,
          inputs: mix(r.inputs, q.inputs),
          contributions: mix(r.contributions, q.contributions),
          bias: r.bias,
          z: lerp(r.z, q.z, u),
          value: lerp(r.value, q.value, u),
        };
      }),
      sensitivity: A.explain.sensitivity.map((row, m) => mix(row, B.explain.sensitivity[m])),
    },
  };
}

// ── Rays for cars without recorded sensors (geometry only) ─────────────────────────────

interface Segment {
  ax: number;
  ay: number;
  bx: number;
  by: number;
}

const segmentCache = new WeakMap<Track, Segment[]>();

function segmentsOf(track: Track): Segment[] {
  let segments = segmentCache.get(track);
  if (!segments) {
    segments = [];
    for (const loop of [track.left, track.right]) {
      for (let i = 0; i < loop.length; i++) {
        const [ax, ay] = loop[i];
        const [bx, by] = loop[(i + 1) % loop.length];
        segments.push({ ax, ay, bx, by });
      }
    }
    segmentCache.set(track, segments);
  }
  return segments;
}

/** Distance from (x, y) along `angle` to the first track edge, capped at `range`. */
export function castRay(track: Track, x: number, y: number, angle: number, range: number): number {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  let best = range;
  for (const s of segmentsOf(track)) {
    const ex = s.bx - s.ax;
    const ey = s.by - s.ay;
    const det = ex * dy - ey * dx;
    if (Math.abs(det) < 1e-9) continue;
    const t = (ex * (s.ay - y) - ey * (s.ax - x)) / det;
    const u = (dx * (s.ay - y) - dy * (s.ax - x)) / det;
    if (t > 0 && t < best && u >= 0 && u <= 1) best = t;
  }
  return best;
}

// ── What-if: the recorded network on changed inputs ──────────────────────────────────────

const ACTIVATIONS: Record<string, (z: number) => number> = {
  tanh: Math.tanh,
  sigmoid: (z) => 1 / (1 + Math.exp(-z)),
  relu: (z) => Math.max(0, z),
  linear: (z) => z,
};

/**
 * An activation function sampled on z from −5 to 5, as `GET activations` returns it. Live, the
 * curve comes from Python; this stands in for it with the recorded source.
 */
export function sampleActivation(name: string): { z: number[]; values: number[] } {
  const f = ACTIVATIONS[name] ?? Math.tanh;
  const z = Array.from({ length: 101 }, (_, i) => -5 + i / 10);
  return { z, values: z.map(f) };
}

/** Forward pass. Production asks Python (`what_if` command); the browser never does this. */
export function forward(network: Network, observation: number[]): number[] {
  let a = observation;
  network.weights.forEach((matrix, layer) => {
    const f = ACTIVATIONS[network.activations[layer]] ?? Math.tanh;
    a = matrix.map((row, j) => f(row.reduce((sum, w, i) => sum + w * a[i], network.biases[layer][j])));
  });
  return a;
}

export interface WhatIfAnswer {
  base: number[];
  tried: number[];
  predicted: number[];
}

export function whatIf(record: CarRecord, frame: CarFrame, overrides: Record<number, number>): WhatIfAnswer {
  const changed = frame.observation.map((v, k) => (k in overrides ? overrides[k] : v));
  const base = frame.explain.receipts.map((r) => r.value);
  const tried = forward(record.network, changed);
  // The straight-line guess: output + Σ sensitivity × change.
  const predicted = base.map((value, m) =>
    frame.explain.sensitivity[m].reduce((sum, s, k) => sum + s * (changed[k] - frame.observation[k]), value),
  );
  return { base, tried, predicted };
}

// ── The what-if ghost: hold an action for 2 s (physics v1, ARCHITECTURE §6.3) ───────────

const PHYSICS = { accel: 8, brake: 16, drag: 0.1, steerMax: 0.5, wheelbase: 2.5, lateralMax: 9, vMax: 30 };

export function rollout(
  pose: { x: number; y: number; psi: number; v: number },
  action: number[],
  seconds: number,
  dt: number,
): { path: number[][]; end: [number, number, number] } {
  let { x, y, psi, v } = pose;
  const [steer, throttle] = action;
  const path: number[][] = [[x, y]];
  for (let k = 0; k < Math.round(seconds / dt); k++) {
    const a = throttle >= 0 ? throttle * PHYSICS.accel : throttle * PHYSICS.brake;
    v = Math.max(0, Math.min(PHYSICS.vMax, v + (a - PHYSICS.drag * v) * dt));
    const wanted = Math.tan(steer * PHYSICS.steerMax) / PHYSICS.wheelbase;
    const limit = PHYSICS.lateralMax / Math.max(v * v, 1);
    const kappa = Math.max(-limit, Math.min(limit, wanted));
    psi -= v * kappa * dt;
    x += v * Math.cos(psi) * dt;
    y += v * Math.sin(psi) * dt;
    path.push([x, y]);
  }
  return { path, end: [x, y, psi] };
}

// ── Flow: what each connection carries right now ─────────────────────────────────────────

/**
 * c[layer][to][from] = weight × the value of the neuron it comes from. Production sends this
 * only for the output layer today (`explain.receipts`); the "Flow" line mode needs it for every
 * layer, which is a small addition to the Python explanation (PRODUCTION_MAPPING.md §6).
 */
export function contributionsOf(network: Network, activations: number[][]): number[][][] {
  return network.weights.map((matrix, layer) => matrix.map((row) => row.map((w, i) => w * (activations[layer]?.[i] ?? 0))));
}
