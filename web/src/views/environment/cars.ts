// How each car looks on the track (task E2, DESIGN_SPEC §7.2): its body colour, the order the cars
// are drawn in, the selected car's name tag, the badge of a stopped car, where the car's drawing
// lands in the world, and where the ray labels go. Pure functions over what the server sent; the
// ranking of a GA race is the GA's own (`population.ts`), nothing here scores a network.

import type { PopulationInfo } from "../../api/generated";
import { carRole, rankBand, rankPosition } from "./population";

/** The size the car sprite is drawn at, metres: 4.5 long (wheelbase 2.5 m + 2 m) and 2.0 wide. */
export const SPRITE_LENGTH = 4.5;
export const SPRITE_WIDTH = 2.0;
/** Pixels per metre from which the full race car is drawn; below it a dart. */
export const DETAIL_SCALE = 4.2;
/** A car is never drawn shorter than this many pixels (only at overview zooms). */
export const MIN_CAR_PIXELS = 10;
/** Up to this many cars, each other driver's car has a livery of its own. */
export const LIVERIES = 10;

/**
 * What a car's body is painted with: its rank band in a GA race (0 = best fifth), a livery
 * (k = 0…9), white when it is alone or the selected one in a crowd, pale grey in a crowd, or the
 * see-through body of a race's ghost.
 */
export type Body =
  | { kind: "rank"; band: number }
  | { kind: "livery"; k: number }
  | { kind: "solo" }
  | { kind: "plain" }
  | { kind: "ghost" };

/**
 * The body of car `i` of `count` cars, `selected` being the selected car. In a race the band is
 * b = min(4, floor(5 · r / (N − 1))) for place r of N (`population.ts`). Otherwise one car is
 * white, up to ten cars take livery i, and more than ten are pale grey except the selected one,
 * which is white: eleven or more colours cannot be told apart.
 */
export function bodyOf(population: PopulationInfo | null, count: number, i: number, selected: number): Body {
  if (population !== null) {
    if (population.ghost_car !== null && i === population.ghost_car) return { kind: "ghost" };
    if (i >= population.size) return { kind: "plain" };
    return { kind: "rank", band: rankBand(rankPosition(population.ranks[i], population.size)) };
  }
  if (count === 1) return { kind: "solo" };
  if (count <= LIVERIES) return { kind: "livery", k: i };
  return i === selected ? { kind: "solo" } : { kind: "plain" };
}

/** The body colour as a CSS value, for HTML and SVG next to the canvas (the legend, E3's panels). */
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
    case "ghost":
      return "var(--car-ghost)";
  }
}

/** The colours a body can take on the canvas (a subset of the scene's `Theme`). */
export interface BodyColours {
  rank: string[];
  livery: string[];
  solo: string;
  plain: string;
  ghostBody: string;
}

/** The body colour as resolved for the canvas. */
export function bodyColor(colours: BodyColours, body: Body): string {
  switch (body.kind) {
    case "rank":
      return colours.rank[body.band];
    case "livery":
      return colours.livery[body.k];
    case "solo":
      return colours.solo;
    case "plain":
      return colours.plain;
    case "ghost":
      return colours.ghostBody;
  }
}

/**
 * The cars of the main pass, in drawing order. In a race the worst rank comes first, so the best
 * end up on top; otherwise by index. The ghost and the selected car are left out: they are drawn
 * after the others, the selected one last, so it is never hidden in a pack.
 * Example: ranks (2, 0, 3, 1), no ghost, car 1 selected → order 2, 0, 3.
 */
export function drawOrder(population: PopulationInfo | null, count: number, selected: number): number[] {
  const order = Array.from({ length: count }, (_, i) => i).filter(
    (i) => i !== selected && (population === null || (i < population.size && i !== population.ghost_car)),
  );
  if (population === null) return order;
  return order.sort((a, b) => population.ranks[b] - population.ranks[a]);
}

/**
 * The selected car's name tag: "YOU" for the keyboard car, "CAR 3" for others, and in a race
 * "CAR 7 · P1 · ELITE · CHAMPION" (place 1 = best) or "GHOST · GEN 4".
 */
export function nameTag(policy: string, population: PopulationInfo | null, selected: number): string {
  if (population === null) return policy === "keyboard" ? "YOU" : `CAR ${selected + 1}`;
  const role = carRole(population, selected);
  if (role === null) return `CAR ${selected + 1}`;
  if (role.kind === "ghost") return ghostTag(population);
  const parts = [`CAR ${selected + 1}`, `P${(role.place ?? 0) + 1}`];
  if (role.elite) parts.push("ELITE");
  if (role.champion) parts.push("CHAMPION");
  return parts.join(" · ");
}

/** The tag beside a race's ghost: "GHOST · GEN 4". */
export function ghostTag(population: PopulationInfo): string {
  return `GHOST · GEN ${population.ghost_unit ?? ""}`;
}

/** The badge colour roles a stopped car can carry. */
export type BadgeColour = "crashed" | "finished" | "stalled" | "timeout";

/**
 * The badge of a stopped car, by `frame.cars.status` (1 crashed, 2 finished, 3 stalled,
 * 4 out of time); running cars (0) have none.
 */
export const BADGES: Record<number, { glyph: string; colour: BadgeColour; label: string }> = {
  1: { glyph: "×", colour: "crashed", label: "crashed" },
  2: { glyph: "✓", colour: "finished", label: "finished" },
  3: { glyph: "Ⅱ", colour: "stalled", label: "stalled" },
  4: { glyph: "◷", colour: "timeout", label: "out of time" },
};

/** Pixels per metre from which stopped cars carry a badge (a dot below) and cars their number. */
export const LABEL_SCALE = 3.2;

/**
 * Whether the cars carry their number on the track (ADR-065): for a driver other than a GA race,
 * with 2 to 10 cars (each in its own livery, which alone cannot tell every pair apart), at a zoom
 * of at least 3.2 px per metre, outside the compact drawer. The selected car has its name tag
 * instead. Example: 8 random networks at 6 px/m → numbers; at 1.2 px/m, or 20 cars → none.
 */
export function showCarNumbers(population: PopulationInfo | null, count: number, scale: number, compact: boolean): boolean {
  return population === null && count >= 2 && count <= LIVERIES && scale >= LABEL_SCALE && !compact;
}

/** True when the car is braking: running and its last throttle below −0.08 (a light touch is not). */
export function isBraking(status: number, action: number[] | undefined): boolean {
  return status === 0 && action !== undefined && (action[1] ?? 0) < -0.08;
}

/**
 * How much a car is enlarged on screen: g = max(1, 10 / (L · s)) for a car L metres long at s
 * pixels per metre, so it is never shorter than 10 px. Example: L = 4.5, s = 1.2 → 5.4 px → g = 1.85.
 */
export function growFactor(length: number, scale: number): number {
  return Math.max(1, MIN_CAR_PIXELS / Math.max(1e-9, length * scale));
}

/**
 * The canvas transform (a, b, c, d, e, f) from the sprite's own frame to world metres: the sprite
 * is drawn with u forward and v to the left for a 4.5 × 2.0 m car centred on (0, 0); it is scaled
 * by g · L / 4.5 along u and g · W / 2.0 along v, turned by the heading ψ and moved to (x, y).
 * A sprite point (u, v) lands at X = a·u + c·v + e, Y = b·u + d·v + f, which for g = 1 and the
 * sprite's own size is X = x + u·cos ψ − v·sin ψ, Y = y + u·sin ψ + v·cos ψ.
 */
export function carTransform(
  x: number,
  y: number,
  psi: number,
  length: number,
  width: number,
  grow = 1,
): [number, number, number, number, number, number] {
  const su = (grow * length) / SPRITE_LENGTH;
  const sv = (grow * width) / SPRITE_WIDTH;
  const c = Math.cos(psi);
  const s = Math.sin(psi);
  return [c * su, s * su, -s * sv, c * sv, x, y];
}

/** Height of a ray label pill and the gap kept between two, CSS pixels. */
export const PILL_HEIGHT = 17;
export const PILL_GAP = 2;
/** How far out along its ray a label is tried, CSS pixels, nearest first. */
export const PILL_PUSHES = [12, 26, 40];

/**
 * Where each ray's label goes (task E2): its pill's left edge x and centre y on screen, or null when
 * it is left out. Symbols: S the car on screen, E_k ray k's end, û = (E_k − S) / |E_k − S| its
 * direction, w_k the pill's width. A pill is tried at E_k + p·û for p = 12, 26, 40 px and kept at
 * the first p where it does not overlap one already placed: two pills overlap when their centres
 * are less than 17 + 2 px apart in y and their x ranges, widened by 2 px, meet. A ray pointing
 * right (ûx > 0.35) has the pill's left edge there, one pointing left (ûx < −0.35) its right edge,
 * one pointing up or down its middle. Order: emphasised rays first, then from the centre ray
 * outwards; in the compact drawer only emphasised rays get a label. With `bounds` (the canvas, w × h
 * px) each tried spot is first moved inside it, 2 px from every edge (Environment gate): x into
 * [2, w − w_k − 2], y into [8.5 + 2, h − 8.5 − 2]. Example: a 46 px pill tried at x = 380 on a
 * 400 px canvas moves to x = 352, so its right edge is at 398.
 */
export function placeRayLabels(
  start: [number, number],
  ends: [number, number][],
  widths: number[],
  highlight: number[],
  compact: boolean,
  bounds?: { w: number; h: number },
): ({ x: number; y: number } | null)[] {
  const placed: { x: number; y: number; w: number }[] = [];
  const overlaps = (x: number, y: number, w: number) =>
    placed.some(
      (b) => x < b.x + b.w + PILL_GAP && b.x < x + w + PILL_GAP && Math.abs(y - b.y) < PILL_HEIGHT + PILL_GAP,
    );
  const centre = (ends.length - 1) / 2;
  const order = ends
    .map((_, k) => k)
    .sort(
      (a, b) =>
        Number(highlight.includes(b)) - Number(highlight.includes(a)) || Math.abs(a - centre) - Math.abs(b - centre),
    );
  const result: ({ x: number; y: number } | null)[] = ends.map(() => null);
  for (const k of order) {
    if (compact && !highlight.includes(k)) continue;
    const [ex, ey] = ends[k];
    const len = Math.hypot(ex - start[0], ey - start[1]) || 1;
    const ux = (ex - start[0]) / len;
    const uy = (ey - start[1]) / len;
    const w = widths[k];
    const shift = ux > 0.35 ? 0 : ux < -0.35 ? -w : -w / 2;
    for (const push of PILL_PUSHES) {
      let x = ex + ux * push + shift;
      let y = ey + uy * push;
      if (bounds !== undefined) {
        x = Math.max(PILL_GAP, Math.min(x, bounds.w - w - PILL_GAP));
        y = Math.max(PILL_HEIGHT / 2 + PILL_GAP, Math.min(y, bounds.h - PILL_HEIGHT / 2 - PILL_GAP));
      }
      if (!overlaps(x, y, w)) {
        placed.push({ x, y, w });
        result[k] = { x, y };
        break;
      }
    }
  }
  return result;
}
