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

// ── a GA network's mutation since its parent (task 7.8, ADR-060) ─────────────────────────────

/** A line of the mutation diff: "pos" / "neg" a mutated weight, "other" a gene from the second parent. */
export interface MutationStroke {
  kind: Sign | "other" | null;
  width: number;
  opacity: number;
}

/** A gene from the other parent that mutation left alone: thin, grey and dashed (see styles.css). */
export const OTHER_PARENT_STROKE: MutationStroke = { kind: "other", width: 0.8, opacity: 0.5 };

/**
 * The line for one weight of the mutation diff: a mutated weight like a checkpoint change
 * (`c` = nudge / largest nudge), a gene taken from the other parent grey, a gene inherited unchanged
 * from the main parent not at all. Mutation wins when a gene is both. Example: c = -1, mutated:
 * an orange line 4.3 px wide; c = 0, from the other parent: the grey line; c = 0 otherwise: nothing.
 */
export function mutationStroke(c: number, mutated: boolean, fromOther: boolean): MutationStroke {
  if (mutated) {
    const s = changeStroke(c);
    // a nudge of exactly 0 (strength 0) still counts as mutated: draw it at the thinnest
    return { kind: s.sign ?? "pos", width: s.sign === null ? 0.6 : s.width, opacity: s.sign === null ? 0.15 : s.opacity };
  }
  return fromOther ? OTHER_PARENT_STROKE : { kind: null, width: 0, opacity: 0 };
}

/** How many weights and biases a mutation diff marks as mutated (only weights are lines). */
export function mutatedCounts(m: {
  weight_mutated: boolean[][][];
  bias_mutated: boolean[][];
}): { weights: number; biases: number } {
  const weights = m.weight_mutated.flat(2).filter(Boolean).length;
  const biases = m.bias_mutated.flat().filter(Boolean).length;
  return { weights, biases };
}

/**
 * One sentence on where a GA network came from. Example: a crossed child of rows 3 and 15 that
 * took 204 of 434 genes from row 15 and had 9 mutated: "child of rows 3 and 15 of generation 11
 * (crossed: 230 genes from row 3, 204 from row 15), 9 genes mutated".
 */
export function originText(m: {
  origin: "elite" | "child";
  unit: number;
  parents: number[];
  crossed: boolean | null;
  main_parent: number;
  other_parent: number | null;
  n_genes: number;
  n_mutated: number;
  n_from_other: number;
}): string {
  const before = m.unit - 1;
  if (m.origin === "elite") return `an exact copy (elite) of row ${m.parents[0]} of generation ${before}: nothing changed`;
  const genes = `${m.n_mutated} gene${m.n_mutated === 1 ? "" : "s"} mutated`;
  const [a, b] = m.parents;
  if (a === b) return `child of row ${a} of generation ${before} with itself, ${genes}`;
  const parents = `child of rows ${a} and ${b} of generation ${before}`;
  if (!m.crossed) return `${parents} (not crossed: a copy of row ${a}), ${genes}`;
  const fromMain = m.n_genes - m.n_from_other;
  return `${parents} (crossed: ${fromMain} genes from row ${m.main_parent}, ${m.n_from_other} from row ${m.other_parent}), ${genes}`;
}
