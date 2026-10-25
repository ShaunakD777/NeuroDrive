import { describe, expect, it } from "vitest";

import type { RunMetrics } from "../../api/generated";
import {
  divergingMix,
  extent,
  formatNumber,
  formatPowerOfTen,
  glowStroke,
  linePath,
  linearScale,
  logScale,
  logTicks,
  lossCurve,
  nearestIndex,
  niceTicks,
  stepDirection,
} from "./charts";

const metrics = (over: Partial<RunMetrics> = {}): RunMetrics => ({
  run_id: "20261025-000000-toy_sine-t",
  unit: [1, 2, 3],
  loss: [0.5, 0.2, 0.1],
  loss_before: [1.0, 0.5, 0.2],
  grad_norm: [[1, 1, 1]],
  diverged: [false, false, false],
  ...over,
});

describe("scales", () => {
  it("maps linearly, including reversed ranges (SVG y grows downwards)", () => {
    const y = linearScale(0, 10, 100, 0);
    expect(y(0)).toBe(100);
    expect(y(10)).toBe(0);
    expect(y(2.5)).toBe(75);
    expect(linearScale(3, 3, 0, 10)(3)).toBe(5);
  });

  it("gives every factor of ten the same distance on a log scale", () => {
    const y = logScale(0.001, 1, 300, 0);
    expect(y(1)).toBeCloseTo(0);
    expect(y(0.001)).toBeCloseTo(300);
    expect(y(0.1) - y(1)).toBeCloseTo(y(0.01) - y(0.1));
  });

  it("finds the extent, skipping nulls and non-finite values", () => {
    expect(extent([3, null, -1, NaN, 2])).toEqual([-1, 3]);
    expect(extent([null])).toEqual([0, 1]);
  });

  it("chooses round linear ticks", () => {
    expect(niceTicks(-3.5, 3.5)).toEqual([-2, 0, 2]);
    expect(niceTicks(-3.5, 3.5, 8)).toEqual([-3, -2, -1, 0, 1, 2, 3]);
    expect(niceTicks(0, 1)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(niceTicks(2, 2)).toEqual([2]);
  });

  it("lists powers of ten that cover the data", () => {
    expect(logTicks(0.0002, 0.3)).toEqual([0.0001, 0.001, 0.01, 0.1, 1]);
    expect(logTicks(0.5, 0.5)).toEqual([0.1, 1]);
  });
});

describe("loss curve", () => {
  it("starts at unit 0 with the untrained loss, then each unit's loss", () => {
    expect(lossCurve(metrics())).toEqual({ unit: [0, 1, 2, 3], loss: [1.0, 0.5, 0.2, 0.1] });
  });

  it("leaves out diverged units", () => {
    const m = metrics({ loss: [0.5, 9.0, null], loss_before: [1.0, 0.5, 9.0], diverged: [false, false, true] });
    expect(lossCurve(m)).toEqual({ unit: [0, 1, 2], loss: [1.0, 0.5, 9.0] });
  });

  it("is empty for a run without metrics", () => {
    expect(lossCurve(metrics({ unit: [], loss: [], loss_before: [], diverged: [] }))).toEqual({
      unit: [],
      loss: [],
    });
  });
});

describe("paths and lookups", () => {
  it("draws a path through the points", () => {
    expect(linePath([0, 10, 20], [5, 6, 7.25])).toBe("M0.0 5.0 L10.0 6.0 L20.0 7.3");
  });

  it("finds the nearest checkpoint", () => {
    const units = [0, 1, 2, 5, 10, 20, 50];
    expect(nearestIndex(units, 0)).toBe(0);
    expect(nearestIndex(units, 7)).toBe(3);
    expect(nearestIndex(units, 8)).toBe(4);
    expect(nearestIndex(units, 999)).toBe(6);
  });
});

describe("colour strengths", () => {
  it("is neutral at the midpoint and full at the class labels", () => {
    expect(divergingMix(0.5, 0, 1)).toEqual({ pole: "mid", percent: 0 });
    expect(divergingMix(1, 0, 1)).toEqual({ pole: "pos", percent: 100 });
    expect(divergingMix(0, 0, 1)).toEqual({ pole: "neg", percent: 100 });
    expect(divergingMix(0.75, 0, 1)).toEqual({ pole: "pos", percent: 50 });
    expect(divergingMix(-3, 0, 1)).toEqual({ pole: "neg", percent: 100 }); // clamped
  });

  it("makes the most-pushed weight bold and an unpushed one faint", () => {
    expect(glowStroke(1)).toEqual({ opacity: 1, width: 4.25 });
    expect(glowStroke(0)).toEqual({ opacity: 0.08, width: 0.75 });
    expect(glowStroke(0.5).opacity).toBeCloseTo(0.54);
    expect(glowStroke(7)).toEqual(glowStroke(1));
  });

  it("knows which way a gradient-descent step moves a weight", () => {
    expect(stepDirection(-0.3)).toBe("pos"); // negative gradient: the weight grows
    expect(stepDirection(0.3)).toBe("neg");
    expect(stepDirection(0)).toBe("mid");
  });

  it("labels log ticks as plain numbers or powers of ten", () => {
    expect(formatPowerOfTen(0.0001)).toBe("1e-4");
    expect(formatPowerOfTen(1e-32)).toBe("1e-32");
    expect(formatPowerOfTen(0.001)).toBe("0.001");
    expect(formatPowerOfTen(10)).toBe("10");
  });

  it("formats numbers compactly", () => {
    expect(formatNumber(0.000218577)).toBe("2.19e-4");
    expect(formatNumber(0.263384)).toBe("0.263");
    expect(formatNumber(1.4575)).toBe("1.46");
    expect(formatNumber(0)).toBe("0");
  });
});
