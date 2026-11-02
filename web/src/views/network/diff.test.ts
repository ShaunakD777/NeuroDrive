import { describe, expect, it } from "vitest";

import { changeStroke, previousCheckpoint } from "./diff";

describe("changeStroke (task 5.9)", () => {
  it("draws the Brief's worked example: change [0.25, 0, -1]", () => {
    const [a, b, c] = [0.25, 0, -1].map(changeStroke);
    expect(a.sign).toBe("pos");
    expect(a.width).toBeCloseTo(1.525, 12);
    expect(a.opacity).toBeCloseTo(0.3625, 12);
    expect(b).toEqual({ sign: null, width: 0, opacity: 0 }); // unchanged: not drawn
    expect(c).toEqual({ sign: "neg", width: 4.3, opacity: 1 }); // the largest change
  });

  it("never draws beyond full strength", () => {
    expect(changeStroke(1.5)).toEqual(changeStroke(1));
  });
});

describe("previousCheckpoint", () => {
  it("finds the checkpoint just before, in any order, and none before the first", () => {
    const units = [0, 1, 2, 3, 5, 10, 20];
    expect(previousCheckpoint(units, 10)).toBe(5);
    expect(previousCheckpoint([20, 0, 5], 20)).toBe(5);
    expect(previousCheckpoint(units, 0)).toBeNull();
  });
});
