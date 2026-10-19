import { create } from "zustand";

import type { SelectedCar } from "../api/generated";
import { useLive } from "./live";

// The selection store: the single place the UI remembers what the user has picked. Every view
// reads from it and writes to it, which is how clicking in one view changes the others
// (docs/ARCHITECTURE.md §8.1-8.2). The server owns which car's details it sends; `selectCar`
// asks it to change, and the store follows whatever car the server's frames say is selected.

export type Mode = "live" | "replay";
export type OutputName = "steer" | "throttle";

/** One neuron: which layer (0 = the inputs), and which position within that layer. */
export interface NeuronRef {
  layer: number;
  index: number;
}

/** One connection (weight): the weight layer it belongs to, the source neuron and the target neuron. */
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
  /**
   * Select car `index` everywhere: remember it and ask the server for that car's details
   * (`select_car`). Does nothing if it is already the selected car. The neuron and connection
   * selections are kept, so the same neuron can be compared across cars.
   */
  selectCar: (index: number) => void;
  /**
   * Select a neuron of a network with `layers` layers (inputs to outputs), or clear it if it is
   * already selected. Selecting a neuron clears the connection; an output neuron also sets
   * `output` (steer for the first output, throttle for the second).
   */
  toggleNeuron: (ref: NeuronRef, layers: number) => void;
  /** Select a connection, or clear it if it is already selected. Clears the neuron. */
  toggleConnection: (ref: ConnectionRef) => void;
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

const OUTPUT_NAMES: OutputName[] = ["steer", "throttle"];

export const useSelection = create<SelectionState & SelectionActions>()((set, get) => ({
  ...initialSelection,
  select: (changes) => set(changes),
  reset: () => set({ ...initialSelection, whatIf: {} }),

  selectCar: (index) => {
    if (get().carId === index) return;
    set({ carId: index });
    useLive.getState().send({ type: "select_car", index });
  },

  toggleNeuron: (ref, layers) => {
    const current = get().neuron;
    if (current !== null && current.layer === ref.layer && current.index === ref.index) {
      set({ neuron: null, output: null });
      return;
    }
    const isOutput = ref.layer === layers - 1;
    set({
      neuron: ref,
      connection: null,
      output: isOutput ? (OUTPUT_NAMES[ref.index] ?? null) : null,
    });
  },

  toggleConnection: (ref) => {
    const c = get().connection;
    if (c !== null && c.layer === ref.layer && c.from === ref.from && c.to === ref.to) {
      set({ connection: null });
      return;
    }
    set({ connection: ref, neuron: null, output: null });
  },
}));

/** Number of rays: inputs 0 to 4 of the observation are the rays, far-left to far-right. */
const RAY_COUNT = 5;

/**
 * Which rays the Environment view should emphasise for the current selection:
 * - neuron driver: the two rays the neuron reads, while the neuron is selected;
 * - network: ray k when input neuron k (k < 5) is selected, or a weight leaving it.
 * Speed and heading error (inputs 5 and 6) have no ray. Anything else highlights nothing.
 */
export function raysFor(
  selection: Pick<SelectionState, "neuron" | "connection">,
  selected: Pick<SelectedCar, "neuron" | "trace"> | null | undefined,
): number[] {
  if (selected === null || selected === undefined) return [];
  if (selected.neuron !== null && selected.neuron !== undefined) {
    return selection.neuron === null ? [] : [...selected.neuron.input_rays];
  }
  if (selected.trace === null || selected.trace === undefined) return [];
  const { neuron, connection } = selection;
  if (neuron !== null && neuron.layer === 0 && neuron.index < RAY_COUNT) return [neuron.index];
  if (connection !== null && connection.layer === 0 && connection.from < RAY_COUNT) {
    return [connection.from];
  }
  return [];
}

// Follow the server: its frames say which car is selected (it starts at car 0 after every
// scenario load), and a new scenario may have a different network, so neuron and connection
// selections from the old one no longer apply.
useLive.subscribe((live, previous) => {
  const selection = useSelection.getState();
  if (live.scenario !== previous.scenario && live.scenario !== null) {
    useSelection.setState({ neuron: null, connection: null, output: null });
  }
  const index = live.frame?.selected?.index;
  if (index !== undefined && index !== selection.carId) {
    useSelection.setState({ carId: index });
  }
});
