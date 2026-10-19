import { describe, expect, it } from "vitest";

import type { Explanation, OutputReceipt } from "../../api/generated";
import {
  applyOverrides,
  describeOutput,
  maxAbs,
  nudgeSentence,
  outputHeadline,
  rankedReceipt,
  raySensitivity,
  rayWidth,
  sliderRange,
  topAndRest,
} from "./inspect";

const receipt: OutputReceipt = {
  output: 0,
  inputs: [0.5, -0.9, 0.2, 0.1],
  contributions: [0.25, -1.8, 0.1, -0.1],
  bias: 0.1,
  z: -1.45,
  value: -0.8957,
};

const explain: Explanation = {
  receipts: [receipt],
  sensitivity: [
    [0.36, -0.36, 0.05, 0, -0.01, 0.2, 0.9],
    [0.01, 0.02, 0.03, 0.04, 0.05, -0.6, 0],
  ],
};

describe("output words", () => {
  it("names the output, its value and its meaning", () => {
    expect(outputHeadline(0, -0.93)).toBe("Steer = −0.93 → hard left");
    expect(outputHeadline(1, 0.4)).toBe("Throttle = 0.40 → accelerating");
    expect(describeOutput(0, 0.01)).toBe("straight");
    expect(describeOutput(1, -0.9)).toBe("hard braking");
  });
});

describe("the receipt", () => {
  it("ranks hidden neurons by the size of their contribution", () => {
    const rows = rankedReceipt(receipt, 2);
    expect(rows.map((r) => r.index)).toEqual([1, 0, 2, 3]); // 0.1 and -0.1 tie: neuron order
    expect(rows[0]).toEqual({ index: 1, label: "Hidden 2, neuron 2", contribution: -1.8 });
  });

  it("shows the top rows and sums the rest", () => {
    const { shown, rest, restCount } = topAndRest(rankedReceipt(receipt, 2), 2);
    expect(shown.map((r) => r.index)).toEqual([1, 0]);
    expect(restCount).toBe(2);
    expect(rest).toBeCloseTo(0); // 0.1 + (-0.1)
    const total = shown.reduce((s, r) => s + r.contribution, 0) + rest + receipt.bias;
    expect(total).toBeCloseTo(receipt.z); // the rows still add up to z
  });
});

describe("sensitivity", () => {
  it("reads a slope as a sentence about a 0.1 nudge", () => {
    expect(nudgeSentence(0, 0, -0.36)).toBe(
      "If Ray +60° rose by 0.1, steer would change by about −0.036 (towards the left).",
    );
    expect(nudgeSentence(1, 5, 0.6)).toBe(
      "If Speed rose by 0.1, throttle would change by about 0.060 (towards accelerating).",
    );
    expect(nudgeSentence(0, 3, 0.001)).toContain("it would barely move");
  });

  it("takes the five ray sensitivities of the chosen output", () => {
    expect(raySensitivity(explain, 0)).toEqual([0.36, -0.36, 0.05, 0, -0.01]);
    expect(raySensitivity(explain, 1)).toEqual([0.01, 0.02, 0.03, 0.04, 0.05]);
    expect(raySensitivity(explain, null)).toBeNull();
    expect(raySensitivity(null, 0)).toBeNull();
  });

  it("scales ray widths by the strongest ray", () => {
    expect(rayWidth(0, 0.36)).toBeCloseTo(1.5);
    expect(rayWidth(-0.36, 0.36)).toBeCloseTo(7);
    expect(rayWidth(0.18, 0.36)).toBeCloseTo(4.25);
    expect(rayWidth(0.5, 0)).toBeCloseTo(7); // no division by zero
    expect(maxAbs([0.1, -0.4, 0.2])).toBe(0.4);
    expect(maxAbs([0, 0])).toBe(1e-6);
  });
});

describe("what-if inputs", () => {
  it("applies only the overridden inputs", () => {
    expect(applyOverrides([0.1, 0.2, 0.3], { 1: 0.9 })).toEqual([0.1, 0.9, 0.3]);
    expect(applyOverrides([0.1, 0.2], {})).toEqual([0.1, 0.2]);
  });

  it("keeps each slider in its input's range", () => {
    expect(sliderRange(0)).toEqual({ min: 0, max: 1 });
    expect(sliderRange(5)).toEqual({ min: 0, max: 1 });
    expect(sliderRange(6)).toEqual({ min: -1, max: 1 });
  });
});
