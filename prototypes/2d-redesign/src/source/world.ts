// The one place the views get the world snapshot from. `startWorld` subscribes to the stores of
// the chosen source and rebuilds the snapshot when one of its inputs changes (and only then, so a
// layout change does not redraw the network). Components use `useWorld`; the canvas, which
// redraws every display frame outside React, uses `getWorld`.

import { create } from "zustand";

import { useLive } from "@prod/state/live";
import { usePlayback } from "@prod/state/playback";
import { useSelection } from "@prod/state/selection";

import { useApp } from "../state/store";
import { buildLive, startLive, useLiveOrigin } from "./live";
import type { SourceKind, World } from "./model";
import { buildRecorded } from "./recorded";

const useWorldStore = create<{ world: World | null }>(() => ({ world: null }));

let previousInputs: readonly unknown[] = [];

/** Rebuild only when an input is a different object or value than last time. */
function changed(inputs: readonly unknown[]): boolean {
  const same = inputs.length === previousInputs.length && inputs.every((v, i) => Object.is(v, previousInputs[i]));
  if (!same) previousInputs = inputs;
  return !same;
}

function recomputeRecorded(): void {
  const s = useApp.getState();
  if (!changed([s.data, s.t, s.car, s.viewKey, s.playing, s.speed, s.whatIf])) return;
  useWorldStore.setState({ world: buildRecorded(s) });
}

function recomputeLive(): void {
  const live = useLive.getState();
  const viewStep = usePlayback.getState().viewStep;
  const overrides = useSelection.getState().whatIf;
  const origin = useLiveOrigin.getState().origin;
  const inputs = [live.scenario, live.geometry, live.frame, live.network, live.whatIfResult, live.trail, live.moment, live.episodeEnd, viewStep, overrides, origin];
  if (!changed(inputs)) return;
  useWorldStore.setState({ world: buildLive({ live, viewStep, overrides, origin }) });
}

/** Start following the chosen source. Returns a function that stops it. */
export function startWorld(source: SourceKind): () => void {
  previousInputs = [];
  if (source === "recorded") {
    recomputeRecorded();
    return useApp.subscribe(recomputeRecorded);
  }
  const stopLive = startLive();
  const stops = [
    useLive.subscribe(recomputeLive),
    usePlayback.subscribe(recomputeLive),
    useSelection.subscribe(recomputeLive),
    useLiveOrigin.subscribe(recomputeLive),
  ];
  recomputeLive();
  return () => {
    stops.forEach((stop) => stop());
    stopLive();
  };
}

/** The current snapshot, for React components (null until the source has a track and a frame). */
export function useWorld(): World | null {
  return useWorldStore((s) => s.world);
}

/** The current snapshot, for code outside React (the canvas loop, event handlers). */
export function getWorld(): World | null {
  return useWorldStore.getState().world;
}
