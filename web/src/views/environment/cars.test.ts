import { describe, expect, it } from "vitest";

import type { PopulationInfo } from "../../api/generated";
import {
  BADGES,
  bodyColor,
  bodyCss,
  bodyOf,
  carTransform,
  drawOrder,
  ghostTag,
  growFactor,
  isBraking,
  nameTag,
  placeRayLabels,
  showCarNumbers,
} from "./cars";
import { carCorners } from "./geometry";

/** population.test.ts's example: F = (0.3, 1.2, 0.3, 1.9), places (2, 1, 3, 0), car 0 elite, 3 champion. */
function race(ghost: boolean): PopulationInfo {
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

describe("car bodies", () => {
  it("paints a race by the GA's rank bands and the ghost see-through", () => {
    const info = race(true);
    const bodies = [0, 1, 2, 3, 4].map((i) => bodyOf(info, 5, i, 0));
    // places (2, 1, 3, 0) of 4 → q = (0.67, 0.33, 1, 0) → bands (3, 1, 4, 0)
    expect(bodies).toEqual([
      { kind: "rank", band: 3 },
      { kind: "rank", band: 1 },
      { kind: "rank", band: 4 },
      { kind: "rank", band: 0 },
      { kind: "ghost" },
    ]);
    expect(bodyCss(bodies[3])).toBe("var(--car-rank-1)"); // the best fifth is --car-rank-1
    expect(bodyCss(bodies[4])).toBe("var(--car-ghost)");
  });

  it("gives other drivers white alone, a livery each up to ten, and grey in a crowd", () => {
    expect(bodyOf(null, 1, 0, 0)).toEqual({ kind: "solo" });
    expect([0, 7, 9].map((i) => bodyOf(null, 10, i, 0))).toEqual([
      { kind: "livery", k: 0 },
      { kind: "livery", k: 7 },
      { kind: "livery", k: 9 },
    ]);
    expect(bodyCss(bodyOf(null, 10, 9, 0))).toBe("var(--car-livery-10)");
    // Eleven or more: pale grey, the selected one white.
    expect(bodyOf(null, 11, 4, 2)).toEqual({ kind: "plain" });
    expect(bodyOf(null, 11, 2, 2)).toEqual({ kind: "solo" });
  });

  it("resolves a body to the theme's colour", () => {
    const colours = {
      rank: ["r1", "r2", "r3", "r4", "r5"],
      livery: Array.from({ length: 10 }, (_, k) => `l${k + 1}`),
      solo: "solo",
      plain: "plain",
      ghostBody: "ghost",
    };
    expect(bodyColor(colours, { kind: "rank", band: 4 })).toBe("r5");
    expect(bodyColor(colours, { kind: "livery", k: 2 })).toBe("l3");
    expect(bodyColor(colours, { kind: "ghost" })).toBe("ghost");
  });
});

describe("draw order", () => {
  it("draws a race worst first and leaves out the ghost and the selected car", () => {
    // ranks (2, 1, 3, 0): worst is car 2 (place 3), then 0, 1, 3. Car 1 selected, ghost 4.
    expect(drawOrder(race(true), 5, 1)).toEqual([2, 0, 3]);
    expect(drawOrder(race(false), 4, 3)).toEqual([2, 0, 1]);
  });

  it("draws other drivers by index without the selected car", () => {
    expect(drawOrder(null, 4, 2)).toEqual([0, 1, 3]);
    expect(drawOrder(null, 1, 0)).toEqual([]);
  });
});

describe("name tags and badges", () => {
  it("names the keyboard car YOU and other cars by number", () => {
    expect(nameTag("keyboard", null, 0)).toBe("YOU");
    expect(nameTag("random_networks", null, 6)).toBe("CAR 7");
  });

  it("names a race car by place and role, and the ghost by its generation", () => {
    const info = race(true);
    expect(nameTag("population", info, 3)).toBe("CAR 4 · P1 · CHAMPION");
    expect(nameTag("population", info, 0)).toBe("CAR 1 · P3 · ELITE");
    expect(nameTag("population", info, 2)).toBe("CAR 3 · P4");
    expect(nameTag("population", info, 4)).toBe("GHOST · GEN 2");
    expect(ghostTag(info)).toBe("GHOST · GEN 2");
  });

  it("has a badge for every stopped status and none while running", () => {
    expect(BADGES[0]).toBeUndefined();
    expect([1, 2, 3, 4].map((s) => BADGES[s]?.colour)).toEqual(["crashed", "finished", "stalled", "timeout"]);
    expect([1, 2, 3, 4].map((s) => BADGES[s]?.glyph)).toEqual(["×", "✓", "Ⅱ", "◷"]);
  });

  it("numbers the cars of 2 to 10 liveries at close zoom only (ADR-065)", () => {
    expect(showCarNumbers(null, 8, 6, false)).toBe(true);
    expect(showCarNumbers(null, 2, 3.2, false)).toBe(true);
    expect(showCarNumbers(null, 10, 6, false)).toBe(true);
    expect(showCarNumbers(null, 1, 6, false)).toBe(false); // one car alone: white, no number
    expect(showCarNumbers(null, 11, 6, false)).toBe(false); // a crowd: grey, no liveries
    expect(showCarNumbers(null, 8, 1.2, false)).toBe(false); // zoomed out
    expect(showCarNumbers(null, 8, 6, true)).toBe(false); // the compact drawer
    expect(showCarNumbers(race(false), 4, 6, false)).toBe(false); // a race: rank colours
  });

  it("lights the brakes below −0.08 throttle on running cars only", () => {
    expect(isBraking(0, [0, -0.3])).toBe(true);
    expect(isBraking(0, [0, -0.05])).toBe(false);
    expect(isBraking(1, [0, -0.3])).toBe(false); // a crashed car is not braking
    expect(isBraking(0, undefined)).toBe(false);
  });
});

describe("where the car's drawing lands", () => {
  /** Apply a canvas transform (a, b, c, d, e, f) to a point. */
  const apply = ([a, b, c, d, e, f]: number[], u: number, v: number) => [a * u + c * v + e, b * u + d * v + f];

  it("puts the Brief's nose and rear-left corner where the formula says", () => {
    const m = carTransform(10, 5, Math.PI / 2, 4.5, 2.0);
    const nose = apply(m, 2.25, 0);
    const rearLeft = apply(m, -2.25, 0.95);
    expect(nose[0]).toBeCloseTo(10, 9);
    expect(nose[1]).toBeCloseTo(7.25, 9);
    expect(rearLeft[0]).toBeCloseTo(9.05, 9);
    expect(rearLeft[1]).toBeCloseTo(2.75, 9);
  });

  it("agrees with carCorners for the scenario's car size", () => {
    for (const [x, y, psi, length, width] of [
      [3, -4, 0.7, 4.5, 2.0],
      [-20, 11, -2.4, 5.1, 1.8], // a different car: the sprite is stretched to it
    ]) {
      const m = carTransform(x, y, psi, length, width);
      const sprite = [
        [2.25, 1.0],
        [2.25, -1.0],
        [-2.25, -1.0],
        [-2.25, 1.0],
      ].map(([u, v]) => apply(m, u, v));
      const expected = carCorners(x, y, psi, length, width);
      sprite.forEach(([px, py], k) => {
        expect(px).toBeCloseTo(expected[k][0], 9);
        expect(py).toBeCloseTo(expected[k][1], 9);
      });
    }
  });

  it("never draws a car shorter than 10 px", () => {
    expect(growFactor(4.5, 1.2)).toBeCloseTo(10 / 5.4, 9); // the Brief: 1.85
    expect(growFactor(4.5, 3)).toBe(1);
    expect(growFactor(4.5, 6)).toBe(1);
    const m = carTransform(0, 0, 0, 4.5, 2.0, growFactor(4.5, 1.2));
    expect((apply(m, 2.25, 0)[0] - apply(m, -2.25, 0)[0]) * 1.2).toBeCloseTo(10, 9);
  });
});

describe("ray label placement", () => {
  const widths = [46, 46, 46];

  it("reproduces the Brief's three rays: the centre first, B left out, C placed at once", () => {
    // index 1 is the centre ray A; 0 is B (10 px below A's end), 2 is C (30 px below).
    const spots = placeRayLabels(
      [0, 0],
      [
        [100, 10],
        [100, 0],
        [100, 30],
      ],
      widths,
      [],
      false,
    );
    expect(spots[1]).toEqual({ x: 112, y: 0 });
    expect(spots[0]).toBeNull();
    expect(spots[2]?.x).toBeCloseTo(100 + 12 * (100 / Math.hypot(100, 30)), 9);
    expect(spots[2]?.y).toBeCloseTo(30 + 12 * (30 / Math.hypot(100, 30)), 9);
  });

  it("places an emphasised ray first, so it keeps its label", () => {
    const spots = placeRayLabels(
      [0, 0],
      [
        [100, 10],
        [100, 0],
        [100, 30],
      ],
      widths,
      [0],
      false,
    );
    expect(spots[0]?.x).toBeCloseTo(100 + 12 * (100 / Math.hypot(100, 10)), 9);
    expect(spots[1]).toBeNull(); // now A is the one pushed out
  });

  it("pushes a pill further out when the first spot is taken", () => {
    // Two rays along the same line, ends 14 px apart: the second tries 12, then 26 px out.
    const spots = placeRayLabels(
      [0, 0],
      [
        [0, 100],
        [0, 114],
      ],
      [30, 30],
      [],
      false,
    );
    // Both point straight down (centred pills); ray 0 (tie, the earlier) goes first at y = 112.
    expect(spots[0]).toEqual({ x: -15, y: 112 });
    // Ray 1 at 12 px: y = 126, 14 apart (overlap); at 26 px: y = 140, 28 apart: free.
    expect(spots[1]).toEqual({ x: -15, y: 140 });
  });

  it("keeps the 2 px gap between pills", () => {
    // Two rays straight down, ends 18 px apart: the first tries are 18 px apart in y, more than a
    // pill's 17 px but less than 17 + 2, so they overlap and the second moves on to 26 px.
    const spots = placeRayLabels(
      [0, 0],
      [
        [0, 100],
        [0, 118],
      ],
      [30, 30],
      [],
      false,
    );
    expect(spots[0]).toEqual({ x: -15, y: 112 });
    expect(spots[1]).toEqual({ x: -15, y: 144 });
  });

  it("puts the pill's right edge at the spot when the ray points left", () => {
    const [spot] = placeRayLabels([0, 0], [[-100, 0]], [40], [], false);
    expect(spot).toEqual({ x: -112 - 40, y: 0 });
  });

  it("labels only emphasised rays in the compact drawer", () => {
    const spots = placeRayLabels([0, 0], [[100, -60], [100, 0], [100, 60]], widths, [2], true);
    expect(spots[0]).toBeNull();
    expect(spots[1]).toBeNull();
    expect(spots[2]).not.toBeNull();
  });
});
