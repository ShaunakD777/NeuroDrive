// Population mode (task 7.6, ADR-057): how a raced GA generation is coloured and labelled. Pure
// functions over what the server sent (the ranks and roles in `scenario.population`); nothing
// here scores or ranks a network: the ranking is the GA's own, computed in Python.

import type { PopulationInfo } from "../../api/generated";

/** The number of colour bands (each one fifth of the ranking; `--rank-1` … `--rank-5`). */
export const RANK_BANDS = 5;

/**
 * Where a place r (0 = best) sits between the best and the worst of N networks:
 * q = r / (N − 1), 0 for the best and 1 for the worst (0 when there is only one network).
 * Example: N = 4 and places (2, 1, 3, 0) give (0.67, 0.33, 1, 0).
 */
export function rankPosition(place: number, size: number): number {
  return size <= 1 ? 0 : place / (size - 1);
}

/**
 * The colour band of position q: band b = min(B − 1, floor(q · B)) with B = 5 bands, 0 the
 * strongest colour (the best fifth) and 4 the palest (the worst fifth). Example: q = 0.67 → 3.
 */
export function rankBand(q: number): number {
  return Math.min(RANK_BANDS - 1, Math.max(0, Math.floor(q * RANK_BANDS)));
}

/** What one car of a race is: a network of the generation (with its role) or the ghost. */
export interface CarRole {
  kind: "network" | "ghost";
  /** The generation's row (car index) for a network, null for the ghost. */
  row: number | null;
  /** Its place in the GA's ranking (0 = best), null for the ghost. */
  place: number | null;
  /** Its colour band (0 = strongest), null for the ghost. */
  band: number | null;
  elite: boolean;
  champion: boolean;
  /** The generation's fitness F of this network (or of the ghost's champion). */
  fitness: number | null;
}

/** The role of car `car` in a race, or null when it is not part of one. */
export function carRole(info: PopulationInfo | null | undefined, car: number): CarRole | null {
  if (info === null || info === undefined) return null;
  if (info.ghost_car !== null && car === info.ghost_car) {
    return { kind: "ghost", row: null, place: null, band: null, elite: false, champion: false, fitness: info.ghost_fitness };
  }
  if (car < 0 || car >= info.size) return null;
  const place = info.ranks[car];
  return {
    kind: "network",
    row: car,
    place,
    band: rankBand(rankPosition(place, info.size)),
    elite: info.elite.includes(car),
    champion: info.champion === car,
    fitness: info.fitness[car],
  };
}

/** A short label for a car: "car 3 · rank 1 of 50 · elite · champion", or the ghost's. */
export function roleLabel(info: PopulationInfo, role: CarRole): string {
  if (role.kind === "ghost") return `ghost: generation ${info.ghost_unit}'s champion`;
  const parts = [`car ${(role.row ?? 0) + 1}`, `rank ${(role.place ?? 0) + 1} of ${info.size}`];
  if (role.elite) parts.push("elite (a copy of the previous generation's best)");
  else if (info.unit > 1) parts.push("child");
  if (role.champion) parts.push("champion");
  return parts.join(" · ");
}

/** The ghost generation the controls offer by default: the previous one (none in generation 1). */
export function defaultGhost(unit: number): number | null {
  return unit > 1 ? unit - 1 : null;
}
