import { signOf, type Sign } from "./networkLayout";

// How the Network view draws what changed between two checkpoints (task 5.9, ADR-042). The
// server sends each weight's change as c = delta / M, between -1 and 1, where delta is the new
// weight minus the old and M the largest |delta| in the network; this only turns c into a line.

export interface ChangeStroke {
  /** "pos": the weight increased (blue), "neg": it decreased (orange), null: unchanged. */
  sign: Sign | null;
  width: number;
  opacity: number;
}

/**
 * The line for a weight whose change is `c`: colour by direction, thickness and strength by |c|
 * (the most-changed weight, |c| = 1, at full strength). An unchanged weight is not drawn, so a
 * checkpoint where learning kept the old network shows an empty diagram.
 * Example: c = 0.25 gives a blue line 1.525 px wide at opacity 0.3625; c = -1 an orange line
 * 4.3 px wide at opacity 1; c = 0 nothing.
 */
export function changeStroke(c: number): ChangeStroke {
  const a = Math.min(1, Math.abs(c));
  if (a === 0) return { sign: null, width: 0, opacity: 0 };
  return { sign: signOf(c), width: 0.6 + 3.7 * a, opacity: 0.15 + 0.85 * a };
}

/** The checkpoint before `unit` among `units` (in any order), or null for the first one. */
export function previousCheckpoint(units: readonly number[], unit: number): number | null {
  let best: number | null = null;
  for (const u of units) if (u < unit && (best === null || u > best)) best = u;
  return best;
}
