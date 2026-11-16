import { describe, expect, it } from "vitest";

import type { AncestryStepView } from "../../api/generated";
import { lineSummary, markOf, stepText } from "./ancestryChart";

const step = (s: Partial<AncestryStepView>): AncestryStepView => ({
  unit: 2,
  row: 1,
  fitness: 2,
  rank: 0,
  origin: "child",
  parents: [3, 1],
  crossed: true,
  main_parent: 3,
  from_main: 1,
  from_other: 1,
  mutated: 1,
  distance: 0.4,
  ...s,
});

// the 7.8 Brief's worked example: generation 3's champion, an elite copy of generation 2's row 1
const LINE = [
  step({ unit: 3, row: 0, origin: "elite", parents: [1], crossed: null, main_parent: 1, mutated: 0 }),
  step({}),
  step({ unit: 1, row: 3, fitness: 1.9, origin: "random", parents: [], crossed: null, main_parent: null, mutated: null }),
];

describe("the ancestry section (task 7.8)", () => {
  it("marks each step by its origin", () => {
    expect(LINE.map(markOf)).toEqual(["elite", "crossed", "random"]);
    expect(markOf({ origin: "child", crossed: false })).toBe("child");
    expect(markOf({ origin: "unknown", crossed: null })).toBe("unknown");
  });

  it("describes the worked example's steps", () => {
    expect(LINE.map(stepText)).toEqual([
      "generation 3, row 0 (rank 1, F 2): an exact copy of row 1",
      "generation 2, row 1 (rank 1, F 2): crossed child of rows 3 and 1, 1 gene from row 1, 1 mutated",
      "generation 1, row 3 (rank 1, F 1.9): a random genome",
    ]);
    expect(stepText(step({ parents: [5, 5], crossed: false, mutated: 2 }))).toContain("child of row 5 with itself, 2 mutated");
  });

  it("sums a line: one elite copy, one crossed child, one mutated gene", () => {
    expect(lineSummary(LINE)).toEqual({ elite: 1, child: 1, crossed: 1, mutated: 1 });
  });
});
