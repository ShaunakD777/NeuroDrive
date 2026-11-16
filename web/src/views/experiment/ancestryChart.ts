import type { AncestryStepView } from "../../api/generated";

// The pure parts of the Ancestry section (task 7.8, ADR-060): how each step of a line of descent
// is marked and described. The line itself is computed by Python (`GET .../ancestry`).

/** How a step is drawn: by its origin, with crossed children apart from uncrossed ones. */
export type Mark = "random" | "elite" | "child" | "crossed" | "unknown";

export const MARKS: { key: Mark; label: string }[] = [
  { key: "random", label: "random (generation 1)" },
  { key: "elite", label: "elite: an exact copy" },
  { key: "child", label: "child, not crossed" },
  { key: "crossed", label: "crossed child" },
  { key: "unknown", label: "origin not recorded" },
];

export function markOf(step: Pick<AncestryStepView, "origin" | "crossed">): Mark {
  if (step.origin === "child") return step.crossed ? "crossed" : "child";
  return step.origin;
}

/**
 * One step in words. Example: a crossed child of rows 3 and 15 that is row 7 of generation 12
 * and took 204 genes from row 15: "generation 12, row 7 (rank 2, F 1.23): crossed child of rows
 * 3 and 15, 204 genes from row 15, 9 mutated".
 */
export function stepText(step: AncestryStepView): string {
  const head = `generation ${step.unit}, row ${step.row} (rank ${step.rank + 1}, F ${step.fitness})`;
  const mutated = `${step.mutated} mutated`;
  switch (step.origin) {
    case "random":
      return `${head}: a random genome`;
    case "unknown":
      return `${head}: origin not recorded (the run was resumed from before task 7.8)`;
    case "elite":
      return `${head}: an exact copy of row ${step.parents[0]}`;
    default: {
      const [a, b] = step.parents;
      if (a === b) return `${head}: child of row ${a} with itself, ${mutated}`;
      const other = step.main_parent === a ? b : a;
      if (!step.crossed) return `${head}: child of rows ${a} and ${b}, a copy of row ${a}, ${mutated}`;
      const n = step.from_other ?? 0;
      return `${head}: crossed child of rows ${a} and ${b}, ${n} gene${n === 1 ? "" : "s"} from row ${other}, ${mutated}`;
    }
  }
}

/** What a line of descent was made of: its steps by origin and the genes mutated along it. */
export function lineSummary(steps: readonly AncestryStepView[]): {
  elite: number;
  child: number;
  crossed: number;
  mutated: number;
} {
  let elite = 0;
  let child = 0;
  let crossed = 0;
  let mutated = 0;
  for (const s of steps) {
    if (s.origin === "elite") elite += 1;
    if (s.origin === "child") {
      child += 1;
      if (s.crossed) crossed += 1;
      mutated += s.mutated ?? 0;
    }
  }
  return { elite, child, crossed, mutated };
}
