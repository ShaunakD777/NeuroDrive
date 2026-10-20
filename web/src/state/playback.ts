import { create } from "zustand";

import type { FrameMessage, MomentMessage } from "../api/generated";
import { useLive } from "./live";
import { useSelection } from "./selection";

// The playback clock (task 2.11, ADR-027): which moment of the selected car's drive the views
// show. `viewStep` null means "now" (the live frames); a number means "look at that earlier step",
// which asks the Python server for the car's state and network trace then (`inspect_step`). The
// simulation itself never moves back: this only changes what is displayed.

export interface PlaybackState {
  /** The earlier step being looked at, or null for the present. */
  viewStep: number | null;
}

export interface PlaybackActions {
  /** Look at step `step` of the selected car (paused only): ask the server for that moment. */
  view: (step: number) => void;
  /** Show the present again. */
  backToLive: () => void;
}

export const usePlayback = create<PlaybackState & PlaybackActions>()((set) => ({
  viewStep: null,
  view: (step) => {
    const frame = useLive.getState().frame;
    if (frame === null || frame.playing) return;
    if (step >= frame.step) {
      set({ viewStep: null }); // the latest step is the present
      return;
    }
    set({ viewStep: step });
    useSelection.getState().clearWhatIf(); // what-if is a question about the present
    useLive.getState().send({ type: "inspect_step", step });
  },
  backToLive: () => set({ viewStep: null }),
}));

/**
 * The moment to show instead of the present, or null: only when a step is being looked at and the
 * server's answer is about that step of the car on screen (an answer for anything else is ignored).
 */
export function activeMoment(
  live: { moment: MomentMessage | null; frame: FrameMessage | null },
  viewStep: number | null,
): MomentMessage | null {
  const { moment, frame } = live;
  if (viewStep === null || moment === null || frame === null) return null;
  if (moment.step !== viewStep || moment.car !== frame.selected?.index) return null;
  return moment;
}

/** The moment being looked at, as a React value (null while showing the present). */
export function useMoment(): MomentMessage | null {
  const moment = useLive((s) => s.moment);
  const frame = useLive((s) => s.frame);
  const viewStep = usePlayback((s) => s.viewStep);
  return activeMoment({ moment, frame }, viewStep);
}

// Back to the present whenever the simulation moves on, another car is chosen or a new scenario
// loads: an old step of another drive or car means nothing.
useLive.subscribe((live, previous) => {
  if (usePlayback.getState().viewStep === null) return;
  const moved =
    live.scenario !== previous.scenario ||
    live.frame === null ||
    (previous.frame !== null &&
      (live.frame.step !== previous.frame.step ||
        live.frame.selected?.index !== previous.frame.selected?.index ||
        live.frame.playing));
  if (moved) usePlayback.setState({ viewStep: null });
});
