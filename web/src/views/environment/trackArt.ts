// Everything decorative about a track, worked out once from the geometry the server already sends
// (`GET tracks/{name}`: centerline, left, right, width, length, start_s0). Nothing here is new data
// and nothing here affects driving: the road the cars and rays "feel" is exactly the two edge
// loops. Kerbs, sand and boards sit OUTSIDE those loops, so the picture never suggests the car has
// room it does not have; the white edge line is drawn exactly on the edge (task E1, lifted from the
// redesign prototype's `trackArt.ts`).
//
// Symbols used below: n centreline samples about Î” = length / n metres apart; h_i the heading of
// the centreline at sample i (radians, from its two neighbours); Îº_i the signed curvature (1/m,
// positive turning left), Îº_i = wrap(h_{i+1} âˆ’ h_{iâˆ’1}) / (2Î”), smoothed over Â±5 samples. 1/Îº is
// the bend's radius.

import type { TrackGeometry } from "../../api/generated";

export type Point = [number, number];

/** One kerb block: four corners, two on the edge and two `KERB_WIDTH` outside it. */
export interface Quad {
  points: Point[];
  /** 0 or 1: which of the two kerb colours (they alternate every `KERB_BLOCK` metres). */
  tone: number;
}

/** A distance board: where it stands (metres, outside the right edge) and the distance it shows. */
export interface Board {
  x: number;
  y: number;
  metres: number;
}

export interface TrackArt {
  /** Arc length of each centreline sample, metres from sample 0. */
  s: number[];
  /** Heading of the centreline at each sample, radians. */
  heading: number[];
  /** Signed curvature at each sample, 1/m: positive turns left. */
  curvature: number[];
  kerbs: Quad[];
  /** Sand traps: closed polygons outside the tight bends. */
  sand: Point[][];
  /** Direction arrows painted on the road: position and heading. */
  arrows: { x: number; y: number; heading: number }[];
  boards: Board[];
  /** The start line: its two ends, the unit tangent along the road, and the sample index. */
  start: { left: Point; right: Point; tangent: Point; index: number };
}

export const KERB_WIDTH = 0.95; // metres, outside the edge
export const KERB_BLOCK = 2.2; // metres of arc length per colour block
export const INSIDE_KERB_RADIUS = 85; // bends tighter than this get a kerb on the inside
export const TIGHT_RADIUS = 46; // tighter than this: also an exit kerb and a sand trap outside
export const SAND_WIDTH = 7.5; // metres, the widest part of a sand trap
export const BOARD_EVERY = 100; // metres between distance boards
export const BOARD_OFFSET = 3.2; // metres outside the right edge
export const ARROW_EVERY = 60; // metres between direction arrows
export const ARROW_RADIUS = 60; // arrows only where the bend is gentler than this

const cache = new WeakMap<TrackGeometry, TrackArt>();

/** An angle wrapped into (âˆ’Ï€, Ï€]. */
function wrap(angle: number): number {
  let a = angle;
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

/** True within `reach` samples of any true entry (the loop wraps round). */
function dilate(mask: boolean[], reach: number): boolean[] {
  const n = mask.length;
  return mask.map((_, i) => {
    for (let d = -reach; d <= reach; d++) if (mask[(i + d + n) % n]) return true;
    return false;
  });
}

/** Maximal runs of true entries, as [first, last] index pairs (a run may wrap past the end). */
function runs(mask: boolean[]): [number, number][] {
  const n = mask.length;
  if (mask.every(Boolean)) return [[0, n - 1]];
  const out: [number, number][] = [];
  let start = mask.findIndex((m, i) => m && !mask[(i - 1 + n) % n]);
  if (start < 0) return out;
  const first = start;
  do {
    let end = start;
    while (mask[(end + 1) % n] && (end + 1) % n !== start) end = (end + 1) % n;
    out.push([start, end]);
    let next = (end + 1) % n;
    while (!mask[next] && next !== first) next = (next + 1) % n;
    start = next;
  } while (start !== first);
  return out;
}

/**
 * The decoration of one track, computed once and cached by the geometry object (the live store
 * keeps one object per loaded track, so the cache is hit on every frame).
 */
export function trackArt(track: TrackGeometry): TrackArt {
  const cached = cache.get(track);
  if (cached) return cached;

  const c = track.centerline;
  const n = c.length;
  const at = (i: number) => c[(i + n) % n];

  const s: number[] = [0];
  for (let i = 1; i < n; i++) s.push(s[i - 1] + Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]));

  const heading = c.map((_, i) => Math.atan2(at(i + 1)[1] - at(i - 1)[1], at(i + 1)[0] - at(i - 1)[0]));
  const step = track.length / n;
  const raw = heading.map((_, i) => wrap(heading[(i + 1) % n] - heading[(i - 1 + n) % n]) / (2 * step));
  // Smooth over Â±5 samples so one kink in the samples does not start a kerb.
  const curvature = raw.map((_, i) => {
    let sum = 0;
    for (let d = -5; d <= 5; d++) sum += raw[(i + d + n) % n];
    return sum / 11;
  });

  /** Unit vector pointing away from the road at sample i on the given side: (E_i âˆ’ C_i) / |E_i âˆ’ C_i|. */
  const outward = (i: number, side: "left" | "right"): Point => {
    const edge = track[side][i];
    const dx = edge[0] - c[i][0];
    const dy = edge[1] - c[i][1];
    const len = Math.hypot(dx, dy) || 1;
    return [dx / len, dy / len];
  };
  /** The point d metres outside the edge at sample i: E_i + dÂ·u. */
  const offset = (i: number, side: "left" | "right", d: number): Point => {
    const [ox, oy] = outward(i, side);
    return [track[side][i][0] + ox * d, track[side][i][1] + oy * d];
  };

  // Which samples get a kerb on which side. Positive curvature turns left: the left edge is inside.
  const insideLeft = dilate(curvature.map((k) => k > 1 / INSIDE_KERB_RADIUS), 6);
  const insideRight = dilate(curvature.map((k) => k < -1 / INSIDE_KERB_RADIUS), 6);
  const tightLeftTurn = curvature.map((k) => k > 1 / TIGHT_RADIUS);
  const tightRightTurn = curvature.map((k) => k < -1 / TIGHT_RADIUS);
  // Exit kerbs: on the outside, starting mid-bend and running a little past it.
  const shift = (mask: boolean[], by: number) => mask.map((_, i) => mask[(i - by + n) % n]);
  const exitRight = dilate(shift(tightLeftTurn, 10), 8);
  const exitLeft = dilate(shift(tightRightTurn, 10), 8);

  const kerbs: Quad[] = [];
  const addKerbs = (mask: boolean[], side: "left" | "right") => {
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      if (!mask[i] || !mask[j]) continue;
      kerbs.push({
        points: [
          track[side][i] as Point,
          track[side][j] as Point,
          offset(j, side, KERB_WIDTH),
          offset(i, side, KERB_WIDTH),
        ],
        tone: Math.floor(s[i] / KERB_BLOCK) % 2,
      });
    }
  };
  addKerbs(insideLeft.map((m, i) => m || exitLeft[i]), "left");
  addKerbs(insideRight.map((m, i) => m || exitRight[i]), "right");

  // Sand traps on the outside of tight bends, tapering in and out.
  const sand: Point[][] = [];
  const addSand = (mask: boolean[], side: "left" | "right") => {
    for (const [a, b] of runs(dilate(mask, 12))) {
      const count = ((b - a + n) % n) + 1;
      if (count < 8) continue;
      const inner: Point[] = [];
      const outer: Point[] = [];
      for (let k = 0; k < count; k++) {
        const i = (a + k) % n;
        const u = Math.min(k, count - 1 - k) / Math.min(14, count / 2);
        const taper = Math.min(1, u) ** 0.7;
        inner.push(offset(i, side, KERB_WIDTH * 0.2));
        outer.push(offset(i, side, KERB_WIDTH + SAND_WIDTH * taper));
      }
      sand.push([...inner, ...outer.reverse()]);
    }
  };
  addSand(tightLeftTurn, "right");
  addSand(tightRightTurn, "left");

  // Direction arrows every 60 m, where the road is fairly straight.
  const arrows: TrackArt["arrows"] = [];
  for (let metres = ARROW_EVERY / 2; metres < track.length; metres += ARROW_EVERY) {
    const i = s.findIndex((v) => v >= metres);
    if (i >= 0 && Math.abs(curvature[i]) < 1 / ARROW_RADIUS) arrows.push({ x: c[i][0], y: c[i][1], heading: heading[i] });
  }

  // Distance boards every 100 m of arc length from the start line, outside the right edge: the
  // board for m metres is at sample (start + round(m / L Â· n)) mod n.
  const boards: Board[] = [];
  const startIndex = Math.round((track.start_s0 / track.length) * n) % n;
  for (let metres = BOARD_EVERY; metres < track.length - 30; metres += BOARD_EVERY) {
    const i = (startIndex + Math.round((metres / track.length) * n)) % n;
    const [x, y] = offset(i, "right", BOARD_OFFSET);
    boards.push({ x, y, metres });
  }

  const art: TrackArt = {
    s,
    heading,
    curvature,
    kerbs,
    sand,
    arrows,
    boards,
    start: {
      left: track.left[startIndex] as Point,
      right: track.right[startIndex] as Point,
      tangent: [Math.cos(heading[startIndex]), Math.sin(heading[startIndex])],
      index: startIndex,
    },
  };
  cache.set(track, art);
  return art;
}

/**
 * Distance from point p to the nearest centreline segment, metres. A point is on the road when this
 * is below half the road width; the tests use it to check that no decoration enters the road.
 */
export function distanceToCentreline(track: TrackGeometry, p: Point): number {
  const c = track.centerline;
  let best = Infinity;
  for (let i = 0; i < c.length; i++) {
    const [ax, ay] = c[i];
    const [bx, by] = c[(i + 1) % c.length];
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / len2));
    best = Math.min(best, Math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)));
  }
  return best;
}
