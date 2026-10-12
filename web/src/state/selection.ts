import { create } from "zustand";

// The selection store: the single place the UI remembers what the user has picked. Every view
// reads from it and writes to it, which is how clicking in one view can change the others
// (docs/ARCHITECTURE.md §8.1). Nothing reads or writes it yet; later tasks wire the views in.

export type Mode = "live" | "replay";
export type OutputName = "steer" | "throttle";

/** One neuron: which layer, and which position within that layer. */
export interface NeuronRef {
  layer: number;
  index: number;
}

/** One connection (weight): the layer it feeds, the source neuron and the target neuron. */
export interface ConnectionRef {
  layer: number;
  from: number;
  to: number;
}

export interface SelectionState {
  mode: Mode;
  runId: string | null;
  /** Which checkpoint of the run. */
  unit: number | null;
  carId: number | null;
  neuron: NeuronRef | null;
  connection: ConnectionRef | null;
  /** Time index within the shown drive. */
  step: number;
  output: OutputName | null;
  /** What-if overrides: observation index -> value typed or dragged by the user. */
  whatIf: Record<number, number>;
}

export interface SelectionActions {
  /** Change some fields and leave the rest as they are. */
  select: (changes: Partial<SelectionState>) => void;
  /** Go back to nothing selected. */
  reset: () => void;
}

export const initialSelection: SelectionState = {
  mode: "live",
  runId: null,
  unit: null,
  carId: null,
  neuron: null,
  connection: null,
  step: 0,
  output: null,
  whatIf: {},
};

export const useSelection = create<SelectionState & SelectionActions>()((set) => ({
  ...initialSelection,
  select: (changes) => set(changes),
  reset: () => set({ ...initialSelection, whatIf: {} }),
}));
