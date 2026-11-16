import { create } from "zustand";

import { detailAt, poseAt, type CarPose } from "../mock/mockServer";
import type { CarFrame, CarRecord, Recorded } from "../mock/types";

// One store for the prototype. In production the same information lives in four stores, and the
// names below say which: `useLive` (frames, scenario), `useSelection` (car, neuron, connection,
// output, what-if), `usePlayback` (the step being looked at) and `useLayout` (page / split,
// drawers). PRODUCTION_MAPPING.md §4 maps every field.

export type ViewId = "environment" | "network" | "experiment";
export type ThemeChoice = "light" | "dark" | "auto";
export type OutputName = "steer" | "throttle";
export type CameraMode = "follow" | "fit" | "free";
export type LineMode = "weights" | "flow" | "change" | "mutation";
export type ExperimentTab = "timeline" | "compare" | "training" | "toylab" | "datasets";

export interface NeuronRef {
  layer: number;
  index: number;
}

export interface ConnectionRef {
  layer: number;
  from: number;
  to: number;
}

interface State {
  data: Recorded | null;
  loadError: string | null;

  // layout (production: useLayout + nav/route.ts)
  view: ViewId;
  layout: "page" | "split";
  drawerOpen: boolean;
  drawerTab: "network" | "standings";
  theme: ThemeChoice;
  /** The design-notes layer (prototype only): numbered markers and their explanations. */
  notes: boolean;

  // playback (production: the server's clock; `viewKey` is usePlayback.viewStep)
  t: number;
  playing: boolean;
  speed: number;
  viewKey: number | null;

  // selection (production: useSelection)
  car: number;
  neuron: NeuronRef | null;
  connection: ConnectionRef | null;
  output: OutputName | null;
  whatIf: Record<number, number>;
  /** A neuron pointed at from outside the diagram (an inspector row): lit while hovered. */
  peek: NeuronRef | null;

  // view options
  cameraMode: CameraMode;
  lineMode: LineMode;
  showRays: boolean;
  showTrail: boolean;
  experimentTab: ExperimentTab;
  generation: number;
  evoStart: number | null;
}

interface Actions {
  setData: (data: Recorded) => void;
  set: (changes: Partial<State>) => void;
  selectCar: (index: number) => void;
  toggleNeuron: (ref: NeuronRef, layers: number) => void;
  toggleConnection: (ref: ConnectionRef) => void;
  setWhatIf: (k: number, value: number) => void;
  clearWhatIf: () => void;
  play: () => void;
  pause: () => void;
  stepOnce: () => void;
  reset: () => void;
  /** Look at an earlier recorded frame of the selected car (paused only), or null for now. */
  lookAt: (key: number | null) => void;
}

const OUTPUTS: OutputName[] = ["steer", "throttle"];

function initialView(): ViewId {
  const name = window.location.hash.replace(/^#\/?/, "");
  return name === "network" || name === "experiment" ? name : "environment";
}

export const useApp = create<State & Actions>()((set, get) => ({
  data: null,
  loadError: null,
  view: initialView(),
  layout: "page",
  drawerOpen: true,
  drawerTab: "network",
  theme: "auto",
  notes: false,
  t: 0,
  playing: false,
  speed: 1,
  viewKey: null,
  car: 0,
  neuron: null,
  connection: null,
  output: null,
  whatIf: {},
  peek: null,
  cameraMode: "follow",
  lineMode: "weights",
  showRays: true,
  showTrail: true,
  experimentTab: "timeline",
  generation: 18,
  evoStart: null,

  setData: (data) =>
    set({ data, car: data.race.scenario.population.champion, generation: data.run.generation, t: 118 }),
  set: (changes) => set(changes),

  selectCar: (index) => {
    if (get().car === index) return;
    set({ car: index, whatIf: {}, viewKey: null, lineMode: get().lineMode === "mutation" ? "weights" : get().lineMode });
  },

  toggleNeuron: (ref, layers) => {
    const current = get().neuron;
    if (current !== null && current.layer === ref.layer && current.index === ref.index) {
      set({ neuron: null, output: null });
      return;
    }
    const isOutput = ref.layer === layers - 1;
    set({ neuron: ref, connection: null, output: isOutput ? (OUTPUTS[ref.index] ?? null) : null });
  },

  toggleConnection: (ref) => {
    const c = get().connection;
    if (c !== null && c.layer === ref.layer && c.from === ref.from && c.to === ref.to) {
      set({ connection: null });
      return;
    }
    set({ connection: ref, neuron: null, output: null });
  },

  setWhatIf: (k, value) => set({ whatIf: { ...get().whatIf, [k]: value } }),
  clearWhatIf: () => set({ whatIf: {} }),

  play: () => {
    const data = get().data;
    if (data === null) return;
    const last = data.race.frames.step.length - 1;
    set({ playing: true, viewKey: null, whatIf: {}, t: get().t >= last ? 0 : get().t });
  },
  // Pausing lands on a recorded frame, so every number on screen belongs to one real step.
  pause: () => set({ playing: false, t: Math.round(get().t) }),
  stepOnce: () => {
    const data = get().data;
    if (data === null) return;
    const last = data.race.frames.step.length - 1;
    set({ playing: false, viewKey: null, whatIf: {}, t: Math.min(last, Math.floor(get().t) + 1) });
  },
  reset: () => set({ playing: false, t: 0, viewKey: null, whatIf: {} }),
  lookAt: (key) => {
    if (get().playing) return;
    set({ viewKey: key === null || key >= Math.round(get().t) ? null : key, whatIf: {} });
  },
}));

// The address hash names the page, as in production (web/src/nav/route.ts).
window.addEventListener("hashchange", () => useApp.setState({ view: initialView() }));

// ── Derived values ────────────────────────────────────────────────────────────────────────

/** The recorded network data of the selected car, or null (only the featured cars were recorded). */
export function useRecord(): CarRecord | null {
  return useApp((s) => s.data?.cars.get(s.car) ?? null);
}

/** The frame position the views show: the step being looked at, else now. */
export function useShownT(): number {
  return useApp((s) => s.viewKey ?? s.t);
}

/** The selected car's sensors, trace and explanation at the shown moment. */
export function useDetail(): CarFrame | null {
  const record = useRecord();
  const t = useShownT();
  return record === null ? null : detailAt(record, t);
}

/** The selected car's pose at the shown moment. */
export function usePose(): CarPose | null {
  const race = useApp((s) => s.data?.race ?? null);
  const car = useApp((s) => s.car);
  const t = useShownT();
  return race === null ? null : poseAt(race, t, car);
}

export const STATUS = [
  { key: "running", label: "Running", kind: "run" },
  { key: "crashed", label: "Crashed", kind: "bad" },
  { key: "finished", label: "Finished", kind: "ok" },
  { key: "stalled", label: "Stalled", kind: "warn" },
  { key: "timeout", label: "Out of time", kind: "idle" },
] as const;

export const INPUT_LABELS = ["Ray +60°", "Ray +30°", "Ray 0°", "Ray −30°", "Ray −60°", "Speed", "Heading error"];
export const INPUT_SHORT = ["+60°", "+30°", "0°", "−30°", "−60°", "v", "ψ err"];
export const OUTPUT_LABELS = ["Steer", "Throttle"];

/** A number with a real minus sign and a fixed number of decimals. */
export function num(x: number, digits = 2): string {
  const text = Math.abs(x).toFixed(digits);
  return x < 0 && Number(text) !== 0 ? `−${text}` : text;
}

/** A signed number: always shows + or −. */
export function signed(x: number, digits = 2): string {
  const text = Math.abs(x).toFixed(digits);
  if (Number(text) === 0) return text;
  return `${x < 0 ? "−" : "+"}${text}`;
}

export function describeSteer(v: number): string {
  const size = Math.abs(v);
  if (size < 0.05) return "straight";
  const side = v > 0 ? "right" : "left";
  return size < 0.3 ? `slightly ${side}` : size < 0.7 ? side : `hard ${side}`;
}

export function describeThrottle(v: number): string {
  if (v >= 0.7) return "full throttle";
  if (v >= 0.05) return "accelerating";
  if (v > -0.05) return "coasting";
  if (v > -0.7) return "braking";
  return "hard braking";
}

/** How the prototype names a car: its role in the generation. */
export function carName(data: Recorded, i: number): string {
  const info = data.race.scenario.population;
  if (i === info.ghost_car) return `Ghost · gen ${info.ghost_unit} champion`;
  return `Car ${i + 1}`;
}

export function carRole(data: Recorded, i: number): string {
  const info = data.race.scenario.population;
  if (i === info.ghost_car) return "ghost";
  if (i === info.champion) return "champion";
  if (info.elite.includes(i)) return "elite";
  return "child";
}
