import { describe, expect, it } from "vitest";

import type { RunMetrics, RunSummary } from "../../api/generated";
import {
  MAX_GROUPS,
  bandPath,
  colorSlots,
  divergingMix,
  extent,
  formatNumber,
  formatPowerOfTen,
  groupRuns,
  glowStroke,
  linePath,
  linearScale,
  logScale,
  logTicks,
  lossCurve,
  nearestIndex,
  shortName,
  sweepSeed,
  niceTicks,
  stepDirection,
  toyRunSummary,
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

describe("toyRunSummary", () => {
  it("reads the config file format of task 4.6b", () => {
    const config = {
      method: "toy",
      task: "sine",
      seed: 0,
      tag: "default",
      network: {
        layer_sizes: [1, 16, 1],
        hidden_activation: "tanh",
        output_activation: "linear",
      },
      training: {
        epochs: 3000,
        loss: "mse",
        batch_size: null,
        val_fraction: 0,
      },
      optimizer: {
        name: "adam",
        learning_rate: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
      },
    };
    expect(toyRunSummary(config)).toEqual({
      task: "sine",
      layerSizes: [1, 16, 1],
      hiddenActivation: "tanh",
      outputActivation: "linear",
      optimizer: "adam",
      learningRate: 0.01,
    });
  });

  it("reads the flat config of Phase 3 toy runs as plain SGD", () => {
    const config = {
      task: "xor",
      layer_sizes: [2, 4, 1],
      hidden_activation: "tanh",
      output_activation: "sigmoid",
      learning_rate: 1.0,
      epochs: 3000,
    };
    expect(toyRunSummary(config)).toMatchObject({
      task: "xor",
      layerSizes: [2, 4, 1],
      optimizer: "sgd",
      learningRate: 1,
    });
  });
});

function run(run_id: string, tag: string, sweep: string | null): RunSummary {
  return {
    run_id,
    method: "toy_sine",
    tag,
    status: "completed",
    started_at: "",
    ended_at: null,
    checkpoints: 1,
    last_unit: 1,
    sweep,
  };
}

describe("compare runs helpers", () => {
  it("colorSlots keeps a group's colour when others come and go", () => {
    const first = colorSlots(["a", "b", "c"], {})!;
    expect(first).toEqual({ a: 0, b: 1, c: 2 });
    const second = colorSlots(["a", "c", "d"], first)!; // b removed, d added
    expect(second).toEqual({ a: 0, c: 2, d: 1 }); // c keeps 2, d takes the free slot 1
  });

  it("colorSlots refuses more groups than colours instead of recycling", () => {
    const ids = Array.from({ length: MAX_GROUPS + 1 }, (_, i) => `g${i}`);
    expect(colorSlots(ids.slice(0, MAX_GROUPS), {})).not.toBeNull();
    expect(colorSlots(ids, {})).toBeNull();
  });

  it("bandPath goes along the upper line and back along the lower one", () => {
    expect(bandPath([0, 10, 20], [5, 6, 7], [1, 2, 3])).toBe(
      "M0.0,1.0L10.0,2.0L20.0,3.0L20.0,7.0L10.0,6.0L0.0,5.0Z",
    );
    expect(bandPath([], [], [])).toBe("");
  });

  it("groupRuns puts a sweep's seeds together, in seed order, at its newest run", () => {
    const sweep = "20261031-120000-toy_sine-adam";
    const runs = [
      run("20261031-120005-toy_sine-adam-s10", "adam-s10", sweep),
      run("20261031-120004-toy_sine-one", "one", null),
      run("20261031-120002-toy_sine-adam-s2", "adam-s2", sweep),
      run("20261031-120001-toy_sine-adam-s0", "adam-s0", sweep),
    ];
    const groups = groupRuns(runs);
    expect(groups.map((g) => g.groupId)).toEqual([
      sweep,
      "20261031-120004-toy_sine-one",
    ]);
    expect(groups[0].runs.map((r) => r.tag)).toEqual([
      "adam-s0",
      "adam-s2",
      "adam-s10",
    ]);
    expect(groups[1].sweep).toBeNull();
  });

  it("shortName drops the time stamp and sweepSeed reads the tag", () => {
    expect(shortName("20261031-120000-toy_sine-adam")).toBe("toy_sine-adam");
    expect(sweepSeed(run("x", "adam-s3", "s"))).toBe(3);
    expect(sweepSeed(run("x", "default", null))).toBeNull();
  });
});

describe("extent on long inputs", () => {
  it("handles 200 000 values (Math.min(...values) would overflow the call stack)", () => {
    const values = Array.from({ length: 200_000 }, (_, i) => (i === 123_456 ? 1e-7 : 1 + (i % 7)));
    expect(extent(values)).toEqual([1e-7, 7]);
  });
});
