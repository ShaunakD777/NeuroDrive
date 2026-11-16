import { describe, expect, it } from "vitest";
import type { TrackGeometry } from "../../api/generated";
import { worldToScreen } from "./geometry";
import { worldTransform } from "./scene";
import { BOARD_OFFSET, distanceToCentreline, type Point, trackArt } from "./trackArt";

const WIDTH = 12;

/**
 * A closed track from a dense loop of centreline points, resampled to `spacing` metres the way the
 * server does (equal arc length), with the left edge WIDTH/2 to the left of the driving direction.
 */
function makeTrack(dense: Point[], spacing = 1, startS0 = 0): TrackGeometry {
  const cum = [0];
  for (let i = 1; i <= dense.length; i++) {
    const a = dense[i - 1];
    const b = dense[i % dense.length];
    cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const length = cum[dense.length];
  const n = Math.round(length / spacing);
  const centerline: Point[] = [];
  let j = 0;
  for (let k = 0; k < n; k++) {
    const target = (k * length) / n;
    while (cum[j + 1] < target) j++;
    const t = (target - cum[j]) / (cum[j + 1] - cum[j] || 1);
    const a = dense[j];
    const b = dense[(j + 1) % dense.length];
    centerline.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
  }
  const left: Point[] = [];
  const right: Point[] = [];
  for (let i = 0; i < n; i++) {
    const p = centerline[(i - 1 + n) % n];
    const q = centerline[(i + 1) % n];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
    const [tx, ty] = [(q[0] - p[0]) / len, (q[1] - p[1]) / len];
    const [c0, c1] = centerline[i];
    left.push([c0 - (ty * WIDTH) / 2, c1 + (tx * WIDTH) / 2]);
    right.push([c0 + (ty * WIDTH) / 2, c1 - (tx * WIDTH) / 2]);
  }
  return { name: "test", split: "train", width: WIDTH, length, start_s0: startS0, centerline, left, right };
}

/** A circle of radius r around the origin, driven anticlockwise (a left turn) or clockwise. */
function circle(r: number, anticlockwise = true): TrackGeometry {
  const m = 4000;
  const dense: Point[] = [];
  for (let i = 0; i < m; i++) {
    const a = ((anticlockwise ? 1 : -1) * 2 * Math.PI * i) / m;
    dense.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return makeTrack(dense);
}

/** Two 200 m straights joined by two 20 m hairpins (centres (0, 0) and (200, 0)), anticlockwise. */
function stadium(): TrackGeometry {
  const dense: Point[] = [];
  const r = 20;
  for (let x = 0; x < 200; x += 0.5) dense.push([x, -r]);
  for (let i = 0; i < 400; i++) {
    const a = -Math.PI / 2 + (Math.PI * i) / 400;
    dense.push([200 + r * Math.cos(a), r * Math.sin(a)]);
  }
  for (let x = 200; x > 0; x -= 0.5) dense.push([x, r]);
  for (let i = 0; i < 400; i++) {
    const a = Math.PI / 2 + (Math.PI * i) / 400;
    dense.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return makeTrack(dense);
}

/**
 * A peanut, r(θ) = 120·(1 + 0.35·cos 2θ), driven anticlockwise: mostly left-hand bends (about 80 m
 * radius at θ = 0), but its waist bends to the right (about 68 m radius at θ = ±90°). An S-bend
 * in each half.
 */
function peanut(): TrackGeometry {
  const dense: Point[] = [];
  const m = 6000;
  for (let i = 0; i < m; i++) {
    const a = (2 * Math.PI * i) / m;
    const r = 120 * (1 + 0.35 * Math.cos(2 * a));
    dense.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return makeTrack(dense);
}

const isOn = (edge: number[][], p: Point) => edge.includes(p);

describe("curvature", () => {
  it("is 1 / radius, positive for a left turn and negative for a right turn", () => {
    // Radius 50 m: κ = 1/50 = 0.02 per metre (the Brief's worked example).
    for (const k of trackArt(circle(50)).curvature) expect(k).toBeCloseTo(0.02, 3);
    for (const k of trackArt(circle(50, false)).curvature) expect(k).toBeCloseTo(-0.02, 3);
  });

  it("changes sign through the peanut's S-bends", () => {
    const { curvature } = trackArt(peanut());
    expect(Math.max(...curvature)).toBeGreaterThan(1 / 85);
    expect(Math.min(...curvature)).toBeLessThan(-1 / 85);
  });
});

describe("kerbs and sand", () => {
  it("put a kerb on the inside of a 50 m left bend only, and no sand (not tighter than 46 m)", () => {
    const track = circle(50);
    const art = trackArt(track);
    expect(art.kerbs.length).toBe(track.centerline.length); // one block per sample, all the way round
    for (const q of art.kerbs) expect(isOn(track.left, q.points[0])).toBe(true);
    expect(art.sand).toHaveLength(0);
  });

  it("mirror for a right bend: the inside is the right edge", () => {
    const track = circle(50, false);
    for (const q of trackArt(track).kerbs) expect(isOn(track.right, q.points[0])).toBe(true);
  });

  it("add an exit kerb and a sand trap on the outside of a 40 m bend", () => {
    const track = circle(40);
    const art = trackArt(track);
    expect(art.kerbs.some((q) => isOn(track.left, q.points[0]))).toBe(true);
    expect(art.kerbs.some((q) => isOn(track.right, q.points[0]))).toBe(true);
    expect(art.sand.length).toBeGreaterThan(0);
    // Outside a left bend is the right edge, farther from the circle's centre than the road.
    for (const polygon of art.sand) for (const [x, y] of polygon) expect(Math.hypot(x, y)).toBeGreaterThan(40 + WIDTH / 2);
  });

  it("leave the middle of the stadium's straights bare, and put sand only at the hairpins", () => {
    const art = trackArt(stadium());
    expect(art.kerbs.length).toBeGreaterThan(0);
    expect(art.sand.length).toBe(2);
    for (const q of art.kerbs) for (const [x] of q.points) expect(x < 25 || x > 175).toBe(true);
    for (const polygon of art.sand) for (const [x] of polygon) expect(x < 15 || x > 185).toBe(true);
  });

  it("put kerbs on both sides of the peanut (left bends and right bends)", () => {
    const track = peanut();
    const art = trackArt(track);
    expect(art.kerbs.some((q) => isOn(track.left, q.points[0]))).toBe(true);
    expect(art.kerbs.some((q) => isOn(track.right, q.points[0]))).toBe(true);
    expect(art.sand).toHaveLength(0); // no bend tighter than 46 m
  });

  it("never put a kerb or sand point inside the road", () => {
    for (const track of [circle(50), circle(40), circle(40, false), stadium(), peanut()]) {
      const art = trackArt(track);
      const points = [...art.kerbs.flatMap((q) => q.points), ...art.sand.flat()];
      expect(points.length).toBeGreaterThan(0);
      for (const p of points) expect(distanceToCentreline(track, p)).toBeGreaterThanOrEqual(WIDTH / 2 - 0.05);
    }
  });

  it("alternate the two kerb colours every 2.2 m of arc length", () => {
    const art = trackArt(circle(50));
    expect(new Set(art.kerbs.map((q) => q.tone))).toEqual(new Set([0, 1]));
    // s = 5 m → floor(5 / 2.2) = 2 → tone 0; s = 7 m → 3 → tone 1 (the Brief's example).
    const at = (metres: number) => art.kerbs[art.s.findIndex((v) => v >= metres)].tone;
    expect(at(5)).toBe(Math.floor(art.s[art.s.findIndex((v) => v >= 5)] / 2.2) % 2);
    expect(at(5)).toBe(0);
    expect(at(7)).toBe(1);
  });
});

describe("arrows, boards and the start line", () => {
  it("paint arrows on the straights and none in a 40 m bend", () => {
    expect(trackArt(stadium()).arrows.length).toBeGreaterThan(0);
    expect(trackArt(circle(40)).arrows).toHaveLength(0);
  });

  it("stand a board every 100 m from the start, 3.2 m outside the right edge", () => {
    const track = circle(100); // 628 m: boards at 100 … 500 (none in the last 30 m)
    const art = trackArt(track);
    expect(art.boards.map((b) => b.metres)).toEqual([100, 200, 300, 400, 500]);
    for (const b of art.boards) {
      expect(distanceToCentreline(track, [b.x, b.y])).toBeCloseTo(WIDTH / 2 + BOARD_OFFSET, 1);
      // Anticlockwise, the right edge is outside: radius 100 + 6 + 3.2.
      expect(Math.hypot(b.x, b.y)).toBeCloseTo(109.2, 1);
    }
    // From a start at s0 = 0, the 100 m board stands at angle 100 / 100 = 1 rad.
    expect(Math.atan2(art.boards[0].y, art.boards[0].x)).toBeCloseTo(1, 1);
  });

  it("count the boards from the start line, not from sample 0", () => {
    const base = circle(100);
    const track = { ...base, start_s0: 157 };
    const art = trackArt(track);
    expect(art.start.index).toBe(Math.round((157 / track.length) * track.centerline.length));
    // The 100 m board is 100 m after the start: arc length 257 m, angle 2.57 rad.
    expect(Math.atan2(art.boards[0].y, art.boards[0].x)).toBeCloseTo(2.57, 1);
    // The start line runs from the left edge to the right edge at the start sample.
    expect(art.start.left).toBe(track.left[art.start.index]);
    expect(art.start.right).toBe(track.right[art.start.index]);
  });

  it("are computed once per geometry object", () => {
    const track = circle(60);
    expect(trackArt(track)).toBe(trackArt(track));
    expect(trackArt({ ...track })).not.toBe(trackArt(track));
  });
});

describe("the world transform", () => {
  it("puts a world point where worldToScreen does (the Brief's worked example)", () => {
    const camera = { cx: 10, cy: 20, scale: 4 };
    const size = { w: 800, h: 600 };
    const [a, b, c, d, e, f] = worldTransform(camera, size);
    expect([a, b, c, d, e, f]).toEqual([4, 0, 0, -4, 360, 380]);
    expect([a * 30 + c * 25 + e, b * 30 + d * 25 + f]).toEqual([480, 280]);
    expect(worldToScreen(camera, size, 30, 25)).toEqual([480, 280]);
  });

  it("agrees with worldToScreen for any camera and point", () => {
    const cases = [
      { camera: { cx: -55.5, cy: 120.25, scale: 0.7 }, size: { w: 1040, h: 616 }, p: [13, -40] },
      { camera: { cx: 300, cy: -12, scale: 23 }, size: { w: 390, h: 300 }, p: [301.5, -10.75] },
    ];
    for (const { camera, size, p } of cases) {
      const [a, b, c, d, e, f] = worldTransform(camera, size);
      const [sx, sy] = worldToScreen(camera, size, p[0], p[1]);
      expect(a * p[0] + c * p[1] + e).toBeCloseTo(sx, 9);
      expect(b * p[0] + d * p[1] + f).toBeCloseTo(sy, 9);
    }
  });
});
