import { describe, expect, it } from "vitest";

import type { EvolutionView } from "../../api/generated";
import {
  bandLabel,
  filledBands,
  finishLegend,
  finishShade,
  fitnessName,
  OUTCOMES,
  outcomeOf,
  panelGeneration,
  shadeColour,
  shadeLegend,
  shadeOf,
  spreadLabels,
  startLabel,
  topEdge,
} from "./evolutionCharts";

const EDGES = Array.from({ length: 41 }, (_, k) => Number((k * 0.05).toFixed(12)));

/** The Brief's worked example as generation 1 (F = 0, 0, 0.12, 0.31, 1.27) and a second generation. */
function example(): EvolutionView {
  const counts = [new Array(40).fill(0), new Array(40).fill(0)];
  counts[0][0] = 2;
  counts[0][2] = 1;
  counts[0][6] = 1;
  counts[0][25] = 1;
  counts[1][1] = 1;
  counts[1][4] = 2;
  counts[1][25] = 2;
  return {
    run_id: "20261114-000000-ga-x",
    population: 5,
    elite: 1,
    unit: [1, 2],
    best_so_far: [1.27, 1.27],
    best: [1.27, 1.27],
    q3: [0.31, 1.27],
    median: [0.12, 0.2],
    mean: [0.34, 0.5],
    q1: [0, 0.2],
    worst: [0, 0.05],
    zero: [2, 0],
    diversity: [4, 2],
    finished_share: [null, null],
    finishers: [null, null],
    champion: [4, 0],
    start: null,
    n_starts: null,
    start_finish_share: null,
    start_mean: null,
    band_width: 0.05,
    edges: EDGES,
    counts,
  };
}

describe("heat map shades", () => {
  it("picks the shade from the share of the generation", () => {
    expect([0.02, 0.05, 0.06, 0.4, 0.41].map(shadeOf)).toEqual([1, 1, 2, 4, 5]);
    expect(shadeOf(0)).toBe(0);
    expect(shadeOf(0.1)).toBe(2);
    expect(shadeOf(0.2)).toBe(3);
    expect(shadeOf(1)).toBe(5);
  });

  it("shades the worked example: band 0 holds 40 %, the others 20 %", () => {
    expect(shadeOf(2 / 5)).toBe(4);
    expect(shadeOf(1 / 5)).toBe(3);
  });

  it("uses the strongest rank token for the most networks", () => {
    expect(shadeColour(5)).toBe("var(--rank-1)");
    expect(shadeColour(1)).toBe("var(--rank-5)");
  });

  it("names the counts each shade stands for, for N = 50", () => {
    expect(shadeLegend(50).map((s) => s.label)).toEqual([
      "up to 5 % (1–2)",
      "up to 10 % (3–5)",
      "up to 20 % (6–10)",
      "up to 40 % (11–20)",
      "more than 40 % (21+)",
    ]);
  });
});

describe("bands and generations", () => {
  it("labels a band by its edges", () => {
    expect(bandLabel(EDGES, 2)).toBe("0.10–0.15");
    expect(bandLabel(EDGES, 25)).toBe("1.25–1.30");
  });

  it("lists the filled bands of a generation", () => {
    expect(filledBands(example(), 0)).toEqual([
      [0, 2],
      [2, 1],
      [6, 1],
      [25, 1],
    ]);
  });

  it("puts the heat map's top at the highest filled band", () => {
    expect(topEdge(example())).toBe(1.3);
  });

  it("follows the slider, generation 1 for unit 0, inside the generations there are", () => {
    const units = [1, 2, 3];
    expect(panelGeneration(2, units)).toBe(2);
    expect(panelGeneration(0, units)).toBe(1);
    expect(panelGeneration(9, units)).toBe(3);
    expect(panelGeneration(null, units)).toBe(3);
    expect(panelGeneration(2, [])).toBeNull();
  });
});

describe("end-of-line labels", () => {
  it("keeps at least the gap between labels and stays inside", () => {
    const out = spreadLabels([{ y: 100 }, { y: 102 }, { y: 101 }], 14, 10, 240);
    expect(out.map((l) => l.y)).toEqual([100, 114, 128]);
    const low = spreadLabels([{ y: 239 }, { y: 240 }], 14, 10, 240);
    expect(low.map((l) => l.y)).toEqual([226, 240]);
  });
});

describe("one start at a time (ADR-059)", () => {
  it("shades a start by the share that finishes it", () => {
    expect([0, 0.02, 0.2, 0.21, 0.4, 0.6, 0.8, 0.81, 1].map(finishShade)).toEqual([0, 1, 1, 2, 2, 3, 4, 5, 5]);
    expect(finishLegend().map((l) => l.label)).toEqual([
      "none",
      "up to 20 %",
      "up to 40 %",
      "up to 60 %",
      "up to 80 %",
      "more than 80 %",
    ]);
  });

  it("names every outcome and keeps an unknown one readable", () => {
    expect(OUTCOMES.map((o) => o.label)).toEqual(["finished", "crashed", "stalled", "out of time"]);
    expect(outcomeOf("crashed").colour).toBe("var(--bad)");
    expect(outcomeOf("running").label).toBe("running");
  });

  it("says where a start is and which fitness is drawn", () => {
    expect(startLabel(2, 0.4, 1.5)).toBe("start 3: 40 % of the lap, 1.5 m left");
    expect(startLabel(0, 0, 0)).toBe("start 1: 0 % of the lap, on the centre line");
    expect(startLabel(4, undefined, undefined)).toBe("start 5");
    expect(fitnessName(null)).toBe("fitness F");
    expect(fitnessName(2)).toBe("fitness f on start 3");
  });
});
