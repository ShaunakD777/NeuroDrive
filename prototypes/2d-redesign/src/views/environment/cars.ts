// How each car is painted (DESIGN_SPEC §7.2), for the canvas and for the panels around it.
// In a GA race the body is the GA's rank band, from production's own band maths
// (web/src/views/environment/population.ts); for every other driver it is a livery.

import { rankBand, rankPosition } from "@prod/views/environment/population";

import { carRole, type Scenario, type World } from "../../source/model";
import type { WorldTheme } from "./scene";

export type Body = { kind: "rank"; band: number } | { kind: "livery"; k: number } | { kind: "solo" } | { kind: "plain" };

/** The body of car `i`: its rank band (0 = best fifth), a livery, white when alone, or pale grey in a crowd. */
export function bodyOf(scenario: Scenario, i: number): Body {
  const info = scenario.population;
  if (info !== null) {
    if (i >= info.size) return { kind: "plain" }; // the ghost
    return { kind: "rank", band: rankBand(rankPosition(info.ranks[i], info.size)) };
  }
  const n = scenario.n_cars;
  if (n === 1) return { kind: "solo" };
  if (n <= 10) return { kind: "livery", k: i };
  return { kind: "plain" };
}

/** The body colour as a CSS value, for HTML and SVG. */
export function bodyCss(body: Body): string {
  switch (body.kind) {
    case "rank":
      return `var(--car-rank-${body.band + 1})`;
    case "livery":
      return `var(--car-livery-${body.k + 1})`;
    case "solo":
      return "var(--car-solo)";
    case "plain":
      return "var(--car-plain)";
  }
}

/** The body colour as resolved for the canvas. */
export function bodyColor(theme: WorldTheme, body: Body): string {
  switch (body.kind) {
    case "rank":
      return theme.rank[body.band];
    case "livery":
      return theme.livery[body.k];
    case "solo":
      return theme.solo;
    case "plain":
      return theme.plain;
  }
}

/** Draw order: in a GA race worst rank first, so the best end up on top; otherwise by index. */
export function drawOrder(scenario: Scenario, count: number): number[] {
  const order = Array.from({ length: count }, (_, i) => i);
  const info = scenario.population;
  if (info === null) return order;
  // The ghost (index size) is drawn on its own, after the generation.
  return order.filter((i) => i < info.size).sort((a, b) => info.ranks[b] - info.ranks[a]);
}

/** The selected car's name tag on the track. */
export function nameTag(world: World): string {
  const { scenario, selected } = world;
  const info = scenario.population;
  const role = carRole(scenario, selected);
  if (info === null) return scenario.policy === "keyboard" ? "YOU" : `CAR ${selected + 1}`;
  if (role === "ghost") return `GHOST · GEN ${info.ghost_unit} CHAMPION`;
  return `CAR ${selected + 1} · P${info.ranks[selected] + 1}${role === "child" ? "" : ` · ${role.toUpperCase()}`}`;
}
