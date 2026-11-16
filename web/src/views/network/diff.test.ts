import { describe, expect, it } from "vitest";

import { changeStroke, mutatedCounts, mutationStroke, originText, OTHER_PARENT_STROKE, previousCheckpoint } from "./diff";

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

describe("mutationStroke (task 7.8)", () => {
  it("draws a mutated weight like a change, a gene from the other parent grey, the rest not at all", () => {
    expect(mutationStroke(-1, true, false)).toEqual({ kind: "neg", width: 4.3, opacity: 1 });
    expect(mutationStroke(0.5, true, true).kind).toBe("pos"); // mutation wins
    expect(mutationStroke(0, false, true)).toEqual(OTHER_PARENT_STROKE);
    expect(mutationStroke(0, false, false)).toEqual({ kind: null, width: 0, opacity: 0 });
    expect(mutationStroke(0, true, false).kind).toBe("pos"); // a zero nudge is still drawn
  });
});

describe("originText and mutatedCounts (task 7.8)", () => {
  const base = { unit: 12, n_genes: 434, n_from_other: 0, other_parent: null, crossed: null, n_mutated: 0 };
  it("names elites, copies and crossed children", () => {
    expect(originText({ ...base, origin: "elite", parents: [4], main_parent: 4 })).toBe(
      "an exact copy (elite) of row 4 of generation 11: nothing changed",
    );
    expect(
      originText({ ...base, origin: "child", parents: [3, 15], main_parent: 3, other_parent: 15, crossed: true, n_from_other: 204, n_mutated: 9 }),
    ).toBe("child of rows 3 and 15 of generation 11 (crossed: 230 genes from row 3, 204 from row 15), 9 genes mutated");
    expect(originText({ ...base, origin: "child", parents: [3, 15], main_parent: 3, other_parent: 15, crossed: false, n_mutated: 1 })).toBe(
      "child of rows 3 and 15 of generation 11 (not crossed: a copy of row 3), 1 gene mutated",
    );
    expect(originText({ ...base, origin: "child", parents: [5, 5], main_parent: 5, crossed: false, n_mutated: 2 })).toBe(
      "child of row 5 of generation 11 with itself, 2 genes mutated",
    );
  });
  it("counts mutated weights and biases", () => {
    const m = { weight_mutated: [[[true, false], [false, true]], [[true]]], bias_mutated: [[false, true], [false]] };
    expect(mutatedCounts(m)).toEqual({ weights: 3, biases: 1 });
  });
});
