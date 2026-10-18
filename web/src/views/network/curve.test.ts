import { describe, expect, it } from "vitest";

import {
  curveBounds,
  curvePoints,
  describeSteer,
  dialAngle,
  offChart,
  plotX,
  plotY,
  RANGE_TEXT,
} from "./curve";

const box = { left: 10, top: 20, width: 200, height: 100 };

describe("chart positions", () => {
  it("maps z from -5..5 across the width and clamps outside it", () => {
    expect(plotX(-5, box)).toBe(10);
    expect(plotX(0, box)).toBe(110);
    expect(plotX(5, box)).toBe(210);
    expect(plotX(50, box)).toBe(210);
    expect(plotX(-50, box)).toBe(10);
    expect(offChart(5)).toBe(false);
    expect(offChart(5.01)).toBe(true);
  });

  it("puts larger outputs higher up (smaller y) and clamps to the chart", () => {
    const bounds = { min: -1, max: 1 };
    expect(plotY(1, bounds, box)).toBe(20);
    expect(plotY(-1, bounds, box)).toBe(120);
    expect(plotY(0, bounds, box)).toBe(70);
    expect(plotY(9, bounds, box)).toBe(20);
  });

  it("pads the curve's range and copes with a flat or empty curve", () => {
    const b = curveBounds([-1, 0, 1]);
    expect(b.min).toBeCloseTo(-1.24, 12);
    expect(b.max).toBeCloseTo(1.24, 12);
    const flat = curveBounds([2, 2, 2]);
    expect(flat.max - flat.min).toBeGreaterThan(1);
    expect(curveBounds([])).toEqual({ min: -1, max: 1 });
  });

  it("builds a polyline from z values and curve values", () => {
    const points = curvePoints([-5, 0, 5], [-1, 0, 1], { min: -1, max: 1 }, box);
    expect(points).toBe("10.0,120.0 110.0,70.0 210.0,20.0");
  });
});

describe("words and the steering dial", () => {
  it("names the steering command", () => {
    expect(describeSteer(0)).toBe("straight");
    expect(describeSteer(0.04)).toBe("straight");
    expect(describeSteer(0.2)).toBe("slightly right");
    expect(describeSteer(-0.2)).toBe("slightly left");
    expect(describeSteer(0.5)).toBe("right");
    expect(describeSteer(-0.5)).toBe("left");
    expect(describeSteer(0.86)).toBe("hard right");
    expect(describeSteer(-0.8)).toBe("hard left");
  });

  it("turns the dial clockwise for right and anticlockwise for left, at most 70 degrees", () => {
    expect(dialAngle(1)).toBe(70);
    expect(dialAngle(-1)).toBe(-70);
    expect(dialAngle(0)).toBe(0);
    expect(dialAngle(0.5)).toBe(35);
    expect(dialAngle(3)).toBe(70);
  });

  it("says in words which activations can output negative numbers", () => {
    expect(RANGE_TEXT.tanh).toContain("−1");
    expect(RANGE_TEXT.sigmoid).toContain("never negative");
    expect(RANGE_TEXT.relu).toContain("never negative");
  });
});
