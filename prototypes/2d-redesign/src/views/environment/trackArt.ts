// Everything decorative about a track, worked out once from the geometry the server already
// sends (`GET tracks/{name}`: centerline, left, right, width, length, start_s0). Nothing here is
// new data and nothing here affects driving: the road the cars and rays "feel" is exactly the two
// edge loops. Kerbs, sand and verges sit OUTSIDE those loops, so the picture never suggests the
// car has room it does not have. The white edge line is drawn exactly on the edge.

import type { Track } from "../../mock/types";

export interface Quad {
  points: [number, number][];
  /** 0 or 1: which of the two kerb colours. */
  tone: number;
}

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
  sand: [number, number][][];
  /** Direction arrows painted on the road: position and heading. */
  arrows: { x: number; y: number; heading: number }[];
  boards: Board[];
  /** The start line: the two ends, the unit tangent and the sample index. */
  start: { left: [number, number]; right: [number, number]; tangent: [number, number]; index: number };
}

const KERB_WIDTH = 0.95; // metres, outside the edge
const KERB_BLOCK = 2.2; // metres per colour block
const INSIDE_KERB_RADIUS = 85; // bends tighter than this get a kerb on the inside
const TIGHT_RADIUS = 46; // bends tighter than this also get an exit kerb and a sand trap outside
const SAND_WIDTH = 7.5;

const cache = new WeakMap<Track, TrackArt>();

function wrap(angle: number): number {
  let a = angle;
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
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

export function trackArt(track: Track): TrackArt {
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
  // Smooth over ±5 samples so one kink in the samples does not start a kerb.
  const curvature = raw.map((_, i) => {
    let sum = 0;
    for (let d = -5; d <= 5; d++) sum += raw[(i + d + n) % n];
    return sum / 11;
  });

  /** Unit vector pointing away from the road at sample i on the given side. */
  const outward = (i: number, side: "left" | "right"): [number, number] => {
    const edge = track[side][i];
    const dx = edge[0] - c[i][0];
    const dy = edge[1] - c[i][1];
    const len = Math.hypot(dx, dy) || 1;
    return [dx / len, dy / len];
  };
  const offset = (i: number, side: "left" | "right", d: number): [number, number] => {
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
        points: [track[side][i] as [number, number], track[side][j] as [number, number], offset(j, side, KERB_WIDTH), offset(i, side, KERB_WIDTH)],
        tone: Math.floor(s[i] / KERB_BLOCK) % 2,
      });
    }
  };
  addKerbs(insideLeft.map((m, i) => m || exitLeft[i]), "left");
  addKerbs(insideRight.map((m, i) => m || exitRight[i]), "right");

  // Sand traps on the outside of tight bends, tapering in and out.
  const sand: [number, number][][] = [];
  const addSand = (mask: boolean[], side: "left" | "right") => {
    for (const [a, b] of runs(dilate(mask, 12))) {
      const count = ((b - a + n) % n) + 1;
      if (count < 8) continue;
      const inner: [number, number][] = [];
      const outer: [number, number][] = [];
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
  for (let metres = 30; metres < track.length; metres += 60) {
    const i = s.findIndex((v) => v >= metres);
    if (i >= 0 && Math.abs(curvature[i]) < 1 / 60) arrows.push({ x: c[i][0], y: c[i][1], heading: heading[i] });
  }

  // Distance boards every 100 m of arc length from the start line, 3 m outside the right edge.
  const boards: Board[] = [];
  const startIndex = Math.round((track.start_s0 / track.length) * n) % n;
  for (let metres = 100; metres < track.length - 30; metres += 100) {
    const i = (startIndex + Math.round((metres / track.length) * n)) % n;
    const [x, y] = offset(i, "right", 3.2);
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
      left: track.left[startIndex] as [number, number],
      right: track.right[startIndex] as [number, number],
      tangent: [Math.cos(heading[startIndex]), Math.sin(heading[startIndex])],
      index: startIndex,
    },
  };
  cache.set(track, art);
  return art;
}
