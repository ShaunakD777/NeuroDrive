import { describe, expect, it } from "vitest";

import type { TrackGeometry } from "../../api/generated";
import { minimapFrame, standingsOrder, statusCounts, stripPaths, stripPeak, stripX } from "./stageLayout";

describe("the look-back strip", () => {
  it("scales each row to its own largest value, at least 0.1", () => {
    expect(stripPeak([0.1, -0.4, 0.3])).toBe(0.4); // the Brief's example
    expect(stripPeak([0, 0, 0])).toBe(0.1);
    expect(stripPeak([])).toBe(0.1);
  });

  it("places the first remembered step at 0 and now at the far end", () => {
    expect(stripX(100, 100, 300, 1000)).toBe(0);
    expect(stripX(200, 100, 300, 1000)).toBe(500);
    expect(stripX(300, 100, 300, 1000)).toBe(1000);
    expect(stripX(5, 5, 5, 1000)).toBe(0); // one step: no division by zero
  });

  it("fills above the line for positive values and below it for negative ones", () => {
    // values (0.1, −0.4, 0.3) scaled by 0.4 around y = 14 with half-height 12:
    // −0.4 fills the lower half fully (y = 26), 0.3 three quarters of the upper half (y = 5).
    const { pos, neg } = stripPaths([0.1, -0.4, 0.3], [0, 10, 20], 14, 12, 0.4);
    expect(pos).toBe("M0.0 14L0.0 11.0L10.0 14.0L20.0 5.0L20.0 14Z");
    expect(neg).toBe("M0.0 14L0.0 14.0L10.0 26.0L20.0 14.0L20.0 14Z");
    expect(stripPaths([], [], 14, 12, 1)).toEqual({ pos: "", neg: "" });
  });
});

describe("standings", () => {
  it("orders cars by progress, ties by car number", () => {
    expect(standingsOrder([10, 250, 30, 250])).toEqual([1, 3, 2, 0]);
  });

  it("counts running, finished, crashed and otherwise stopped cars", () => {
    expect(statusCounts([0, 1, 2, 3, 4, 0, 1])).toEqual({ running: 2, finished: 1, crashed: 2, stopped: 2 });
  });
});

describe("the minimap's frame", () => {
  const geometry = {
    name: "square",
    width: 2,
    length: 40,
    start_s0: 0,
    centerline: [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
    ],
    left: [
      [-1, -1],
      [11, -1],
      [11, 11],
      [-1, 11],
    ],
    right: [
      [1, 1],
      [9, 1],
      [9, 9],
      [1, 9],
    ],
  } as unknown as TrackGeometry;

  it("puts the outline's corner at the origin with north up, once per track", () => {
    const frame = minimapFrame(geometry);
    expect(frame).toMatchObject({ minX: -1, maxY: 11, w: 12, h: 12 });
    // (0, 0) is drawn at (0 − (−1), 11 − 0) = (1, 11): bottom left, as on the canvas.
    expect(frame.road.startsWith("M1.0 11.0L11.0 11.0")).toBe(true);
    expect(minimapFrame(geometry)).toBe(frame);
  });
});
