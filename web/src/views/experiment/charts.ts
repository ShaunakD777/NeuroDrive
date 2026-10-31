// Pure helpers for the toy lab's charts: scales, paths, ticks and colour strengths. No React and
// no maths about networks here: every number comes from the Python server (ADR-012); this only maps
// numbers to pixels and styles. See docs/ARCHITECTURE.md §8.

import type { RunMetrics, RunSummary } from "../../api/generated";

export type Scale = (value: number) => number;

/** Straight-line map from [d0, d1] to [r0, r1] (a zero-width domain maps to the middle). */
export function linearScale(d0: number, d1: number, r0: number, r1: number): Scale {
  if (d1 === d0) return () => (r0 + r1) / 2;
  const k = (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

/** Map on a base-10 log scale: equal pixel steps for each factor of 10. Needs d0, d1 > 0. */
export function logScale(d0: number, d1: number, r0: number, r1: number): Scale {
  const inner = linearScale(Math.log10(d0), Math.log10(d1), r0, r1);
  return (v) => inner(Math.log10(Math.max(v, Number.MIN_VALUE)));
}

/** Smallest and largest finite value (null and NaN skipped); [0, 1] when there are none. */
export function extent(values: readonly (number | null)[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : [0, 1];
}

/** The powers of ten from below `lo` to above `hi`, e.g. 0.0002..0.3 → 0.0001, 0.001, 0.01, 0.1, 1. */
export function logTicks(lo: number, hi: number): number[] {
  const first = Math.floor(Math.log10(lo));
  const last = Math.ceil(Math.log10(hi));
  const ticks: number[] = [];
  for (let p = first; p <= Math.max(last, first + 1); p++) ticks.push(10 ** p);
  return ticks;
}

/**
 * About `count` round tick values covering [lo, hi]: steps of 1, 2 or 5 times a power of ten,
 * e.g. -3.5..3.5 → -4, -2, 0, 2, 4 (only ticks inside [lo, hi] are returned).
 */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? 10 * power;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    ticks.push(Number(t.toPrecision(12)));
  }
  return ticks;
}

/**
 * The loss curve to draw: unit 0 is the untrained network (its loss is unit 1's `loss_before`),
 * then every unit's loss after its step. Diverged units (null loss) are left out.
 */
export function lossCurve(m: RunMetrics): { unit: number[]; loss: number[] } {
  const unit: number[] = [];
  const loss: number[] = [];
  if (m.unit.length > 0 && m.loss_before[0] !== null) {
    unit.push(0);
    loss.push(m.loss_before[0]);
  }
  m.unit.forEach((u, i) => {
    const l = m.loss[i];
    if (l !== null) {
      unit.push(u);
      loss.push(l);
    }
  });
  return { unit, loss };
}

/** An SVG path through the points (x[i], y[i]): "M x0 y0 L x1 y1 ...". */
export function linePath(xs: readonly number[], ys: readonly number[]): string {
  return xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
}

/** Index of the entry of ascending `sorted` closest to `value`. */
export function nearestIndex(sorted: readonly number[], value: number): number {
  let lo = 0;
  let hi = sorted.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= value) lo = mid;
    else hi = mid;
  }
  return Math.abs(sorted[hi] - value) < Math.abs(sorted[lo] - value) ? hi : lo;
}

export type Pole = "pos" | "neg" | "mid";

/**
 * Diverging colour for a value between `lo` and `hi` (the two class labels, e.g. 0 and 1): the
 * midpoint is neutral, values towards `hi` take the positive hue and towards `lo` the negative
 * hue. `percent` (0..100) is how much of the hue to mix into the neutral; values beyond the ends
 * are clamped.
 */
export function divergingMix(value: number, lo: number, hi: number): { pole: Pole; percent: number } {
  const mid = (lo + hi) / 2;
  const half = (hi - lo) / 2;
  if (half <= 0) return { pole: "mid", percent: 0 };
  const t = Math.max(-1, Math.min(1, (value - mid) / half));
  if (Math.abs(t) < 1e-9) return { pole: "mid", percent: 0 };
  return { pole: t > 0 ? "pos" : "neg", percent: Math.round(Math.abs(t) * 100) };
}

/**
 * How a connection is drawn from its glow g = |dL/dw| / max|dL/dW| (0..1, from the server):
 * faint and thin when learning barely pushes it, bold when it is the most-pushed weight.
 */
export function glowStroke(glow: number): { opacity: number; width: number } {
  const g = Math.max(0, Math.min(1, glow));
  return { opacity: 0.08 + 0.92 * g, width: 0.75 + 3.5 * g };
}

/**
 * Which way the next gradient-descent step moves a weight: the step is −η·dL/dw, so a negative
 * gradient makes the weight grow ("pos") and a positive one makes it shrink ("neg").
 */
export function stepDirection(gradient: number): Pole {
  if (gradient < 0) return "pos";
  if (gradient > 0) return "neg";
  return "mid";
}

/** A log-axis tick label: 10^p written plainly from 0.001 up, as "1e-4" below that. */
export function formatPowerOfTen(t: number): string {
  const p = Math.round(Math.log10(t));
  return p < -3 ? `1e${p}` : formatNumber(t);
}

/** A number in at most 3 significant digits, with an exponent for very small or large ones. */
export function formatNumber(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
  return Number(v.toPrecision(3)).toString();
}

/** What the toy lab's summary line shows about a run's config. */
export interface ToyRunSummary {
  task: string;
  layerSizes: number[];
  hiddenActivation: string;
  outputActivation: string;
  optimizer: string;
  learningRate: number;
}

/**
 * Read a run's stored config (`GET /api/runs/{id}`) for the summary line. Run folders hold either
 * the config file format of task 4.6b (`network`, `training` and `optimizer` sections, ADR-032)
 * or the flat dictionary of Phase 3 toy runs (always plain SGD). Only reads; computes nothing.
 */
export function toyRunSummary(config: Record<string, unknown>): ToyRunSummary {
  const network = config.network as Record<string, unknown> | undefined;
  const optimizer = config.optimizer as Record<string, unknown> | undefined;
  if (network && optimizer) {
    return {
      task: String(config.task),
      layerSizes: network.layer_sizes as number[],
      hiddenActivation: String(network.hidden_activation),
      outputActivation: String(network.output_activation),
      optimizer: String(optimizer.name),
      learningRate: Number(optimizer.learning_rate),
    };
  }
  return {
    task: String(config.task),
    layerSizes: config.layer_sizes as number[],
    hiddenActivation: String(config.hidden_activation),
    outputActivation: String(config.output_activation),
    optimizer: String(config.optimizer ?? "sgd"),
    learningRate: Number(config.learning_rate),
  };
}

// ---- Compare runs (task 4.7b, ADR-033) ----

/** How many groups can be compared at once: one per categorical colour slot, never recycled. */
export const MAX_GROUPS = 8;

/**
 * Give each group a colour slot (0 to MAX_GROUPS - 1). A group keeps the slot it had before (the
 * colour follows the group, not its position), new groups take the smallest free slot. Returns
 * null when there are more groups than slots.
 */
export function colorSlots(
  groupIds: readonly string[],
  previous: Readonly<Record<string, number>>,
): Record<string, number> | null {
  if (groupIds.length > MAX_GROUPS) return null;
  const slots: Record<string, number> = {};
  const used = new Set<number>();
  for (const id of groupIds) {
    const slot = previous[id];
    if (slot !== undefined && !used.has(slot)) {
      slots[id] = slot;
      used.add(slot);
    }
  }
  for (const id of groupIds) {
    if (slots[id] !== undefined) continue;
    let slot = 0;
    while (used.has(slot)) slot += 1;
    slots[id] = slot;
    used.add(slot);
  }
  return slots;
}

/**
 * The outline of a band between a lower and an upper line through the same x positions (already
 * in pixels): along the upper line left to right, then back along the lower line, closed.
 */
export function bandPath(
  xs: readonly number[],
  lower: readonly number[],
  upper: readonly number[],
): string {
  if (xs.length === 0) return "";
  const top = xs.map((x, i) => `${x.toFixed(1)},${upper[i].toFixed(1)}`);
  const bottom = xs
    .map((x, i) => `${x.toFixed(1)},${lower[i].toFixed(1)}`)
    .reverse();
  return `M${top.join("L")}L${bottom.join("L")}Z`;
}

/** A run-list entry: a sweep with its runs, or a single run (sweep null). */
export interface RunGroup {
  groupId: string;
  sweep: string | null;
  runs: RunSummary[];
}

/**
 * Group the run list (newest first) by sweep: every sweep becomes one entry holding its runs in
 * seed order, at the place of its newest run; a run without a sweep is an entry of its own.
 */
export function groupRuns(runs: readonly RunSummary[]): RunGroup[] {
  const groups: RunGroup[] = [];
  const bySweep = new Map<string, RunGroup>();
  for (const run of runs) {
    const sweep = run.sweep ?? null;
    if (sweep === null) {
      groups.push({ groupId: run.run_id, sweep: null, runs: [run] });
      continue;
    }
    let group = bySweep.get(sweep);
    if (!group) {
      group = { groupId: sweep, sweep, runs: [] };
      bySweep.set(sweep, group);
      groups.push(group);
    }
    group.runs.push(run);
  }
  for (const g of groups)
    g.runs.sort((a, b) => (sweepSeed(a) ?? 0) - (sweepSeed(b) ?? 0));
  return groups;
}

/** The seed of a sweep run, read from its tag `<tag>-s<seed>` (null if the tag has no suffix). */
export function sweepSeed(run: RunSummary): number | null {
  const match = /-s(\d+)$/.exec(run.tag);
  return match ? Number(match[1]) : null;
}

/** A short name for a run or sweep id: without the leading `YYYYMMDD-HHMMSS-` time stamp. */
export function shortName(id: string): string {
  return id.replace(/^\d{8}-\d{6}-/, "");
}
