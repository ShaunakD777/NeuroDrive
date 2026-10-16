import { describe, expect, it } from "vitest";

import {
  carCorners,
  clampScale,
  fitCamera,
  followCamera,
  MAX_SCALE,
  MIN_SCALE,
  nearestCar,
  rayEnd,
  screenToWorld,
  worldToScreen,
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
