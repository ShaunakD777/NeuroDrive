import { describe, expect, it } from "vitest";

import type { PopulationInfo } from "../../api/generated";
import { carRole, defaultGhost, rankBand, rankPosition, RANK_BANDS, roleLabel } from "./population";

/** The Brief's worked example: F = (0.3, 1.2, 0.3, 1.9), so the places are (2, 1, 3, 0). */
function example(ghost: boolean): PopulationInfo {
  return {
    run: "20261113-000000-ga-x",
    unit: 3,
    last_unit: 5,
    start: 0,
    n_starts: 10,
    start_s0_m: 0,
    start_lateral_m: 0,
    size: 4,
    fitness: [0.3, 1.2, 0.3, 1.9],
    ranks: [2, 1, 3, 0],
    finished: [0, 0, 0, 1],
    crashed: [1, 1, 1, 0],
    elite: [0],
    champion: 3,
    ghost_unit: ghost ? 2 : null,
    ghost_car: ghost ? 4 : null,
    ghost_fitness: ghost ? 1.5 : null,
  };
}

describe("rank colours", () => {
  it("places the worked example between best (0) and worst (1)", () => {
    const q = [2, 1, 3, 0].map((r) => rankPosition(r, 4));
    expect(q.map((v) => Number(v.toFixed(2)))).toEqual([0.67, 0.33, 1, 0]);
    expect(q.map(rankBand)).toEqual([3, 1, 4, 0]);
    expect(rankPosition(0, 1)).toBe(0); // a single network is the best
  });

  it("splits 50 networks into five bands of ten", () => {
    const bands = Array.from({ length: 50 }, (_, r) => rankBand(rankPosition(r, 50)));
    for (let b = 0; b < RANK_BANDS; b++) expect(bands.filter((x) => x === b)).toHaveLength(10);
    expect(bands[0]).toBe(0);
    expect(bands[49]).toBe(4);
    expect(bands).toEqual([...bands].sort((a, b) => a - b)); // never stronger for a worse place
  });
});

describe("car roles", () => {
  it("describes every network and the ghost", () => {
    const info = example(true);
    const best = { kind: "network", row: 3, place: 0, band: 0, elite: false, champion: true, fitness: 1.9 };
    expect(carRole(info, 3)).toEqual(best);
    expect(carRole(info, 0)?.elite).toBe(true);
    expect(carRole(info, 2)?.band).toBe(4);
    expect(carRole(info, 4)).toMatchObject({ kind: "ghost", fitness: 1.5, band: null });
    expect(carRole(info, 5)).toBeNull();
    expect(carRole(example(false), 4)).toBeNull();
    expect(carRole(null, 0)).toBeNull();
  });

  it("labels them in words", () => {
    const info = example(true);
    expect(roleLabel(info, carRole(info, 3)!)).toBe("car 4 · rank 1 of 4 · child · champion");
    expect(roleLabel(info, carRole(info, 0)!)).toBe(
      "car 1 · rank 3 of 4 · elite (a copy of the previous generation's best)",
    );
    expect(roleLabel(info, carRole(info, 4)!)).toBe("ghost: generation 2's champion");
    const first = { ...example(false), unit: 1, elite: [] };
    expect(roleLabel(first, carRole(first, 1)!)).toBe("car 2 · rank 2 of 4"); // no parents yet
  });

  it("offers the previous generation as the ghost", () => {
    expect(defaultGhost(1)).toBeNull();
    expect(defaultGhost(7)).toBe(6);
  });
});
