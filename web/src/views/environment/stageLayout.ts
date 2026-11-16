// Pure helpers for the stage around the track (task E3): the look-back strip's scales and paths,
// the standings' order and counts, and the minimap's frame. They only place numbers the server
// sent (the remembered actions, each car's progress and status, the track's outline).

import type { TrackGeometry } from "../../api/generated";

/**
 * A row's scale on the look-back strip: m = max(0.1, max |vᵢ|), so each row fills its height with
 * its own largest value in the window (printed beside the row) and small steering still shows;
 * 0.1 keeps a row of zeros flat instead of blowing up noise. Example: (0.1, −0.4, 0.3) → 0.4.
 */
export function stripPeak(values: number[]): number {
  return values.reduce((m, v) => Math.max(m, Math.abs(v)), 0.1);
}

/** Step k's x on a strip W wide that runs from step `first` (x = 0) to `last` (x = W). */
export function stripX(step: number, first: number, last: number, width: number): number {
  return ((step - first) / Math.max(1, last - first)) * width;
}

/**
 * One row of the strip as two filled areas around its zero line at y = `mid`: the positive part
 * max(v, 0) above it and the negative part min(v, 0) below, value `top` reaching `half` units from
 * the line (SVG y grows downwards). Two paths whatever the number of steps.
 */
export function stripPaths(
  values: number[],
  xs: number[],
  mid: number,
  half: number,
  top: number,
): { pos: string; neg: string } {
  if (values.length === 0) return { pos: "", neg: "" };
  const y = (v: number) => (mid - (v / top) * half).toFixed(1);
  const x = (i: number) => xs[i].toFixed(1);
  let pos = `M${x(0)} ${mid}`;
  let neg = pos;
  values.forEach((v, i) => {
    pos += `L${x(i)} ${y(Math.max(0, v))}`;
    neg += `L${x(i)} ${y(Math.min(0, v))}`;
  });
  const end = `L${x(values.length - 1)} ${mid}Z`;
  return { pos: pos + end, neg: neg + end };
}

/** The cars in race order: furthest along first; equal progress keeps the car order. */
export function standingsOrder(progress: number[]): number[] {
  return progress.map((_, i) => i).sort((a, b) => progress[b] - progress[a] || a - b);
}

/** How many cars are running, finished, crashed, or stopped otherwise (stalled or out of time). */
export function statusCounts(status: number[]): { running: number; finished: number; crashed: number; stopped: number } {
  const counts = { running: 0, finished: 0, crashed: 0, stopped: 0 };
  for (const s of status) {
    if (s === 0) counts.running += 1;
    else if (s === 1) counts.crashed += 1;
    else if (s === 2) counts.finished += 1;
    else counts.stopped += 1;
  }
  return counts;
}

/** The minimap's frame: the outline's box and the centreline as an SVG path (y flipped). */
export interface MinimapFrame {
  minX: number;
  maxY: number;
  w: number;
  h: number;
  road: string;
}

const frames = new WeakMap<TrackGeometry, MinimapFrame>();

/**
 * The minimap's frame of a track, computed once per geometry object: a world point (x, y) is drawn
 * at (x − minX, maxY − y), so north stays up.
 */
export function minimapFrame(geometry: TrackGeometry): MinimapFrame {
  const cached = frames.get(geometry);
  if (cached !== undefined) return cached;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of [...geometry.left, ...geometry.right]) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const road = `M${geometry.centerline.map(([x, y]) => `${(x - minX).toFixed(1)} ${(maxY - y).toFixed(1)}`).join("L")}Z`;
  const frame = { minX, maxY, w: maxX - minX, h: maxY - minY, road };
  frames.set(geometry, frame);
  return frame;
}
