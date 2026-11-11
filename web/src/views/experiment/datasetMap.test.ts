import { describe, expect, it } from "vitest";

import { barSpan, colourScale, denseStretches, formatShare, nearestPoint, parseHex, pointColour, sharedColourScale } from "./datasetMap";

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

describe("sharedColourScale", () => {
  it("covers the largest |value| of both datasets", () => {
    expect(sharedColourScale([0.1, -0.287], [0.6, -0.2])).toBe(0.6);
    expect(sharedColourScale([0.1, -0.287], [])).toBe(0.29);
    expect(sharedColourScale([0], [0])).toBe(1);
  });
});

describe("barSpan", () => {
  it("puts paired bars side by side inside the bin", () => {
    expect(barSpan(100, 141, 0, 2)).toEqual([101, 120]);
    expect(barSpan(100, 141, 1, 2)).toEqual([121, 140]);
    expect(barSpan(100, 141, 0, 1)).toEqual([101, 140]); // one bar: the 6.2 layout
  });
});

describe("denseStretches", () => {
  it("joins the dense bins and clips them to [min, max]", () => {
    expect(denseStretches([0, 0.25, 0.5, 0.75, 1], [3, 2, 0, 1], 1.5, 0.1, 0.9)).toEqual([[0.1, 0.5]]);
  });
  it("counts a bin exactly at the threshold, and keeps separate stretches apart", () => {
    // The follow-up Brief's worked example: only [0, 0.25) clears 1.5 frames, clipped it is 0.1..0.25.
    expect(denseStretches([0, 0.25, 0.5, 0.75, 1], [3, 1, 0, 1], 1.5, 0.1, 0.9)).toEqual([[0.1, 0.25]]);
    expect(denseStretches([0, 0.25, 0.5, 0.75, 1], [2, 0, 0, 2], 2, 0.1, 0.9)).toEqual([
      [0.1, 0.25],
      [0.75, 0.9],
    ]);
  });
  it("is empty for an empty dataset", () => {
    expect(denseStretches([0, 0.5, 1], [0, 0], 0, null, null)).toEqual([]);
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
