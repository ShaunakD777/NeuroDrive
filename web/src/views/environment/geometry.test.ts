import { describe, expect, it } from "vitest";

import {
  carCorners,
  clampScale,
  fitCamera,
  followCamera,
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
