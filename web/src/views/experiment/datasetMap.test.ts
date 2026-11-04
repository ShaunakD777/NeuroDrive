import { describe, expect, it } from "vitest";

import { colourScale, formatShare, nearestPoint, parseHex, pointColour } from "./datasetMap";

const palette = { pos: [0, 0, 200] as [number, number, number], neg: [200, 100, 0] as [number, number, number], mid: [100, 100, 100] as [number, number, number] };

describe("parseHex", () => {
  it("reads long and short hex colours", () => {
    expect(parseHex("#2563eb")).toEqual([37, 99, 235]);
    expect(parseHex(" #fff ")).toEqual([255, 255, 255]);
    expect(parseHex("rgb(1, 2, 3)")).toBeNull();
  });
});

describe("pointColour", () => {
  it("is the midpoint at 0 and the poles at the ends", () => {
    expect(pointColour(0, -1, 1, palette)).toBe("rgb(100, 100, 100)");
    expect(pointColour(1, -1, 1, palette)).toBe("rgb(0, 0, 200)"); // right
    expect(pointColour(-1, -1, 1, palette)).toBe("rgb(200, 100, 0)"); // left
  });
  it("mixes in proportion: steer 0.5 is half way to the positive pole", () => {
    expect(pointColour(0.5, -1, 1, palette)).toBe("rgb(50, 50, 150)");
  });
});

describe("colourScale", () => {
  it("is the largest |value| rounded up to 2 decimals", () => {
    expect(colourScale([0.1, -0.287, 0.2])).toBe(0.29);
    expect(colourScale([1, -0.81])).toBe(1);
    expect(colourScale([0.25])).toBe(0.25);
    expect(colourScale([0, 0])).toBe(1);
  });
});

describe("nearestPoint", () => {
  it("finds the closest point within the radius, or none", () => {
    const xs = [0, 10, 20];
    const ys = [0, 0, 0];
    expect(nearestPoint(xs, ys, 12, 1)).toBe(1);
    expect(nearestPoint(xs, ys, 15, 30)).toBeNull();
  });
});

describe("formatShare", () => {
  it("writes shares as percentages", () => {
    expect(formatShare(0.034454)).toBe("3.4 %");
    expect(formatShare(0.8819)).toBe("88 %");
    expect(formatShare(0.0003)).toBe("0.03 %");
    expect(formatShare(0)).toBe("0 %");
  });
});
