import { useEffect, useState } from "react";

import type { CheckpointDiff, CheckpointRef, MutationView, PopulationInfo } from "../../api/generated";
import { fetchCheckpointDiff, fetchCheckpoints, fetchMutation } from "../../api/rest";
import { previousCheckpoint } from "./diff";

// The Network view's two server lookups, moved out of NetworkView.tsx unchanged (task S0) so the
// redesigned inspector rail, the breadcrumb and the Timeline can ask the same questions.

/**
 * For a replayed checkpoint: the checkpoint before it and what changed since, both from the server
 * (`GET checkpoints`, `GET checkpoints/{unit}/diff`). `from` is null for a run's first checkpoint.
 */
export function useCheckpointChange(checkpoint: CheckpointRef | null): {
  from: number | null;
  diff: CheckpointDiff | null;
} {
  const [found, setFound] = useState<{ key: string; from: number | null; diff: CheckpointDiff | null } | null>(null);
  const run = checkpoint?.run ?? null;
  const unit = checkpoint?.unit ?? null;
  useEffect(() => {
    if (run === null || unit === null) return;
    let live = true;
    const key = `${run}/${unit}`;
    fetchCheckpoints(run)
      .then(async (list) => {
        const from = previousCheckpoint(list.units, unit);
        const diff = from === null ? null : await fetchCheckpointDiff(run, unit, from);
        if (live) setFound({ key, from, diff });
      })
      .catch(() => live && setFound({ key, from: null, diff: null }));
    return () => {
      live = false;
    };
  }, [run, unit]);
  if (found === null || found.key !== `${run}/${unit}`) return { from: null, diff: null };
  return { from: found.from, diff: found.diff };
}

export type MutationState = { mutation: MutationView | null; note: string | null; error: string | null };

/**
 * In population mode (task 7.6): what made the selected car's network differ from its parents
 * (`GET .../generations/{g}/rows/{r}/mutation`, task 7.8). Generation 1 and the ghost have none;
 * a run trained before task 7.8 gets the server's refusal as `error`.
 */
export function useMutation(population: PopulationInfo | null, car: number | null): MutationState {
  const [found, setFound] = useState<{ key: string; state: MutationState } | null>(null);
  const run = population?.run ?? null;
  const unit = population?.unit ?? null;
  const isGhost = population !== null && car !== null && car >= population.size;
  const key = `${run}/${unit}/${car}`;
  useEffect(() => {
    if (run === null || unit === null || car === null || isGhost || unit < 2) return;
    let live = true;
    fetchMutation(run, unit, car)
      .then((m) => live && setFound({ key, state: { mutation: m, note: null, error: null } }))
      .catch((e: Error) => live && setFound({ key, state: { mutation: null, note: null, error: e.message } }));
    return () => {
      live = false;
    };
  }, [run, unit, car, isGhost, key]);
  if (population === null || car === null) return { mutation: null, note: null, error: null };
  if (isGhost)
    return { mutation: null, note: `the ghost: generation ${population.ghost_unit}'s champion, not part of this generation`, error: null };
  if (population.unit < 2) return { mutation: null, note: "a random genome of generation 1: it has no parent", error: null };
  if (found === null || found.key !== key) return { mutation: null, note: "looking up its parents…", error: null };
  return found.state;
}
