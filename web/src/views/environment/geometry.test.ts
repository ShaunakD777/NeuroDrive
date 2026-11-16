import { describe, expect, it } from "vitest";

import {
  carCorners,
  clampMargins,
  clampScale,
  easeCamera,
  easeOut,
  fitCamera,
  followCamera,
  lookAhead,
  marginShift,
  MAX_SCALE,
  MIN_SCALE,
  nearestCar,
  panBy,
  rayEnd,
  scaleLimits,
  screenToWorld,
  wheelFactor,
  worldToScreen,
  zoomAt,
} from "./geometry";

const size = { w: 200, h: 100 };

describe("world to screen", () => {
  const cam = { cx: 10, cy: 5, scale: 2 };

  it("puts the camera centre in the middle of the canvas", () => {
    expect(worldToScreen(cam, size, 10, 5)).toEqual([100, 50]);
  });

  it("flips y: one metre up is two pixels up the screen at scale 2", () => {
    const [px, py] = worldToScreen(cam, size, 11, 6);
    expect(px).toBe(102);
    expect(py).toBe(48);
  });

  it("round-trips through screenToWorld", () => {
    const [px, py] = worldToScreen(cam, size, -3.5, 17.25);
    const [x, y] = screenToWorld(cam, size, px, py);
    expect(x).toBeCloseTo(-3.5, 12);
    expect(y).toBeCloseTo(17.25, 12);
  });
});

describe("cameras", () => {
  it("follows a point at the requested scale, clamped to the allowed range", () => {
    expect(followCamera(3, 4, 6)).toEqual({ cx: 3, cy: 4, scale: 6 });
    expect(followCamera(0, 0, 1000).scale).toBe(MAX_SCALE);
    expect(clampScale(0)).toBe(MIN_SCALE);
  });

  it("fits a bounding box: centred, and the limiting direction decides the scale", () => {
    // Points span 100 m wide and 20 m tall; the canvas is 200 x 100 with a 20 px margin.
    const cam = fitCamera(
      [
        [0, 0],
        [100, 20],
      ],
      size,
      20,
    );
    expect(cam.cx).toBe(50);
    expect(cam.cy).toBe(10);
    // Width allows (200 - 40) / 100 = 1.6 px/m; height allows (100 - 40) / 20 = 3 px/m.
    expect(cam.scale).toBeCloseTo(1.6, 12);
  });

  it("falls back to a sensible camera with no points or no canvas", () => {
    expect(fitCamera([], size)).toEqual({ cx: 0, cy: 0, scale: 1 });
    expect(fitCamera([[1, 1]], { w: 0, h: 0 })).toEqual({ cx: 0, cy: 0, scale: 1 });
  });
});

describe("rays and cars", () => {
  it("ends a ray of 10 m at the right place for the heading and the ray angle", () => {
    // Car at the origin facing +y (90 degrees); the ray points 90 degrees left of that: towards -x.
    const [x, y] = rayEnd(0, 0, Math.PI / 2, Math.PI / 2, 10);
    expect(x).toBeCloseTo(-10, 12);
    expect(y).toBeCloseTo(0, 12);
    const [fx, fy] = rayEnd(1, 2, 0, 0, 5);
    expect([fx, fy]).toEqual([6, 2]);
  });

  it("gives car corners front-left, front-right, rear-right, rear-left", () => {
    // 4 m long, 2 m wide, at (10, 0) facing +x: front is at x = 12, left is +y.
    const corners = carCorners(10, 0, 0, 4, 2);
    expect(corners[0]).toEqual([12, 1]);
    expect(corners[1]).toEqual([12, -1]);
    expect(corners[2]).toEqual([8, -1]);
    expect(corners[3]).toEqual([8, 1]);
  });

  it("rotates the corners with the heading", () => {
    // Facing +y: the front is above the centre and the left side is towards -x.
    const [frontLeft] = carCorners(0, 0, Math.PI / 2, 4, 2);
    expect(frontLeft[0]).toBeCloseTo(-1, 12);
    expect(frontLeft[1]).toBeCloseTo(2, 12);
  });
});

describe("picking a car", () => {
  const cam = { cx: 0, cy: 0, scale: 10 };
  const big = { w: 400, h: 400 };

  it("returns the closest car within the radius, or null", () => {
    const xs = [0, 5];
    const ys = [0, 0];
    expect(nearestCar(cam, big, xs, ys, 203, 198, 10)).toBe(0);
    expect(nearestCar(cam, big, xs, ys, 251, 200, 10)).toBe(1);
    expect(nearestCar(cam, big, xs, ys, 300, 300, 10)).toBeNull();
    expect(nearestCar(cam, big, [], [], 200, 200, 10)).toBeNull();
  });
});

describe("mouse pan and zoom (task 4.10)", () => {
  const square = { w: 200, h: 200 };
  const limits = { min: 0.5, max: 40 };

  it("zooms around the cursor: the Brief's worked example", () => {
    const cam = zoomAt({ cx: 0, cy: 0, scale: 2 }, square, 150, 100, 2, limits);
    expect(cam).toEqual({ cx: 12.5, cy: 0, scale: 4 });
    expect(worldToScreen(cam, square, 25, 0)).toEqual([150, 100]); // still under the cursor
  });

  it("keeps the point under the cursor for any factor and position", () => {
    const start = { cx: 3, cy: -7, scale: 5 };
    for (const [px, py, f] of [[10, 190, 0.86], [123, 45, 1.16], [200, 0, 3]]) {
      const before = screenToWorld(start, square, px, py);
      const cam = zoomAt(start, square, px, py, f, limits);
      const [sx, sy] = worldToScreen(cam, square, before[0], before[1]);
      expect(sx).toBeCloseTo(px, 9);
      expect(sy).toBeCloseTo(py, 9);
    }
  });

  it("stops at the limits", () => {
    expect(zoomAt({ cx: 0, cy: 0, scale: 30 }, square, 100, 100, 10, limits).scale).toBe(40);
    expect(zoomAt({ cx: 0, cy: 0, scale: 1 }, square, 100, 100, 0.01, limits).scale).toBe(0.5);
  });

  it("pans with the mouse: the Brief's example", () => {
    expect(panBy({ cx: 0, cy: 0, scale: 4 }, 40, -20)).toEqual({ cx: -10, cy: -5, scale: 4 });
  });

  it("turns wheel steps into zoom factors", () => {
    expect(wheelFactor(-100)).toBeCloseTo(1.1618, 4);
    expect(wheelFactor(100)).toBeCloseTo(0.8607, 4);
    expect(wheelFactor(0)).toBe(1);
  });

  it("lowers the zoom-out limit when the whole track needs a smaller scale", () => {
    expect(scaleLimits(2)).toEqual({ min: MIN_SCALE, max: MAX_SCALE });
    expect(scaleLimits(0.3)).toEqual({ min: 0.15, max: MAX_SCALE });
  });
});

describe("camera motion and fit margins (task E3)", () => {
  it("fits into the area the panels leave free: the Brief's example", () => {
    const margins = { top: 70, right: 28, bottom: 164, left: 28 };
    const cam = fitCamera(
      [
        [0, 0],
        [400, 200],
      ],
      { w: 1000, h: 600 },
      margins,
    );
    expect(cam.scale).toBeCloseTo(366 / 200, 12); // 1.83: the height decides
    expect(cam.cx).toBeCloseTo(200, 12); // equal left and right margins
    expect(cam.cy).toBeCloseTo(100 + (70 - 164) / (2 * (366 / 200)), 12); // 74.3
    // The box's top edge lands exactly on the top margin, its bottom edge on the bottom one.
    expect(worldToScreen(cam, { w: 1000, h: 600 }, 0, 200)[1]).toBeCloseTo(70, 9);
    expect(worldToScreen(cam, { w: 1000, h: 600 }, 0, 0)[1]).toBeCloseTo(600 - 164, 9);
    // Uneven left and right margins: the left edge lands on the left margin.
    const lopsided = fitCamera([[0, 0], [400, 200]], { w: 1000, h: 600 }, { top: 0, right: 200, bottom: 0, left: 0 });
    expect(worldToScreen(lopsided, { w: 1000, h: 600 }, 0, 0)[0]).toBeCloseTo(0, 9);
    expect(worldToScreen(lopsided, { w: 1000, h: 600 }, 400, 0)[0]).toBeCloseTo(800, 9);
  });

  it("shifts a followed point into the middle of the free area", () => {
    const [dx, dy] = marginShift({ top: 70, right: 28, bottom: 164, left: 28 }, 6);
    expect(dx).toBeCloseTo(0, 12);
    expect(dy).toBeCloseTo(-94 / 12, 12); // −7.83 m: the camera looks below the car
    // The car at (0, 0) then shows 47 px above the centre of a 1000 × 600 canvas.
    const cam = { cx: dx, cy: dy, scale: 6 };
    expect(worldToScreen(cam, { w: 1000, h: 600 }, 0, 0)[1]).toBeCloseTo(300 - 47, 9);
    expect(marginShift(24, 6)).toEqual([-0, 0]);
  });

  it("is today's fit when every side has the same margin", () => {
    const points = [
      [0, 0],
      [100, 20],
    ];
    expect(fitCamera(points, size, { top: 20, right: 20, bottom: 20, left: 20 })).toEqual(fitCamera(points, size, 20));
  });

  it("eases out and moves the scale geometrically: the Brief's numbers", () => {
    expect(easeOut(0)).toBe(0);
    expect(easeOut(0.5)).toBeCloseTo(0.875, 12);
    expect(easeOut(1)).toBe(1);
    expect(easeOut(2)).toBe(1); // clamped
    const from = { cx: 0, cy: 0, scale: 2 };
    const to = { cx: 100, cy: 0, scale: 8 };
    expect(easeCamera(from, to, 0.5)).toEqual({ cx: 50, cy: 0, scale: 4 });
    const half = easeCamera(from, to, easeOut(0.5));
    expect(half.cx).toBeCloseTo(87.5, 12);
    expect(half.scale).toBeCloseTo(6.727, 3);
    expect(easeCamera(from, to, 1)).toEqual(to);
  });

  it("looks ahead of the car, more the faster it goes, never off the screen", () => {
    const canvas = { w: 1000, h: 600 };
    const [x, y] = lookAhead(100, 50, 0, 20, 6, canvas);
    expect(x).toBeCloseTo(107.6, 12); // 0.28 × 20 + 2 = 7.6 m ahead
    expect(y).toBeCloseTo(50, 12);
    // Paused: 2 m at any zoom, so zooming while following leaves the centre where it was.
    expect(lookAhead(0, 0, Math.PI / 2, 0, 6, canvas)[1]).toBeCloseTo(2, 12);
    expect(lookAhead(0, 0, Math.PI / 2, 0, 20, canvas)[1]).toBeCloseTo(2, 12);
    // Fast and zoomed in: capped at 22 % of the shorter side, 0.22 × 600 / 40 = 3.3 m.
    expect(lookAhead(0, 0, 0, 30, 40, canvas)[0]).toBeCloseTo(3.3, 12);
  });

  it("shrinks the panels' margins to half of a small stage, and leaves a big one alone", () => {
    const small = { w: 390, h: 260 };
    const m = clampMargins({ top: 70, right: 28, bottom: 230, left: 28 }, small);
    expect(m.top).toBeCloseTo((70 * 130) / 300, 9); // 30.3
    expect(m.bottom).toBeCloseTo((230 * 130) / 300, 9); // 99.7
    expect(m.top + m.bottom).toBeCloseTo(130, 9); // half the height stays clear
    expect([m.left, m.right]).toEqual([28, 28]); // 56 ≤ 195: untouched
    const big = { top: 70, right: 28, bottom: 164, left: 28 };
    expect(clampMargins(big, { w: 1000, h: 600 })).toEqual(big);
    expect(clampMargins(24, { w: 1000, h: 600 })).toEqual({ top: 24, right: 24, bottom: 24, left: 24 });
  });

  it("keeps a fast followed car on a small stage under tall panels (E3 verification)", () => {
    // The tests' All layout: a 390 × 260 stage, bottom panels 400 px tall, zoom 18.3, the car at
    // 30 m/s heading down the screen, so the look-ahead and the margin shift both push it up.
    const stage = { w: 390, h: 260 };
    const panels = { top: 70, right: 28, bottom: 400, left: 28 };
    const screenY = (margins: { top: number; right: number; bottom: number; left: number }) => {
      const [lx, ly] = lookAhead(0, 0, -Math.PI / 2, 30, 18.3, stage);
      const [dx, dy] = marginShift(margins, 18.3);
      return worldToScreen(followCamera(lx + dx, ly + dy, 18.3), stage, 0, 0)[1];
    };
    expect(screenY(panels)).toBeLessThan(0); // unclamped: drawn above the canvas
    const y = screenY(clampMargins(panels, stage));
    expect(y).toBeGreaterThan(0);
    expect(y).toBeLessThan(stage.h);
  });
});
