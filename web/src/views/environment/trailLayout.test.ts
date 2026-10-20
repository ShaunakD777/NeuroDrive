import { describe, expect, it } from "vitest";

import { agoText, nearestTrailStep, trailStyle, trailValues } from "./trailLayout";

const trail = {
  car: 0,
  steps: [10, 11, 12],
  x: [0, 1, 2],
  y: [0, 0, 0],
  action: [
    [-0.8, 0.2],
    [0, 0.4],
    [0.5, -1],
  ],
};

describe("trail colours", () => {
  it("take the selected output's value at each point, or nothing when no output is selected", () => {
    expect(trailValues(trail, 0)).toEqual([-0.8, 0, 0.5]);
    expect(trailValues(trail, 1)).toEqual([0.2, 0.4, -1]);
    expect(trailValues(trail, null)).toBeNull();
    expect(trailValues(null, 0)).toBeNull();
  });

  it("sign picks the colour and size the strength (like the rays)", () => {
    expect(trailStyle(0.5).positive).toBe(true);
    expect(trailStyle(-0.5).positive).toBe(false);
    expect(trailStyle(0).width).toBe(1.5);
    expect(trailStyle(-1).width).toBe(5.5);
    expect(trailStyle(3).width).toBe(5.5); // capped at full strength
    expect(trailStyle(1).alpha).toBe(1);
    expect(trailStyle(0.5).width).toBeGreaterThan(trailStyle(0.2).width);
  });
});

describe("clicking the trail", () => {
  const screen: [number, number][] = [
    [100, 100],
    [120, 100],
    [140, 100],
  ];
  it("finds the nearest point within the radius", () => {
    expect(nearestTrailStep(screen, trail.steps, 118, 103, 12)).toBe(11);
    expect(nearestTrailStep(screen, trail.steps, 141, 99, 12)).toBe(12);
  });
  it("finds nothing farther away", () => {
    expect(nearestTrailStep(screen, trail.steps, 120, 130, 12)).toBeNull();
    expect(nearestTrailStep([], [], 0, 0, 12)).toBeNull();
  });
});

it("says how long ago a step was", () => {
  expect(agoText(60, 1 / 30)).toBe("2.0 s ago");
});
