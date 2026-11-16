import { create } from "zustand";

import type { Recorded } from "../mock/types";
import type { SourceKind } from "../source/model";

// The prototype's own store: layout, view options, and (with the recorded source) the playback
// clock and the selection. With the live source the selection and the clock belong to the
// production stores (`useSelection`, `usePlayback`, `useLive` in web/src/state/); the actions
// below then forward to them through the bridge that src/source/live.ts attaches, and the
// selection fields are kept as a mirror of `useSelection` so every view reads them the same way.
// PRODUCTION_MAPPING.md §4 maps every field to its production home.

export type ViewId = "environment" | "network" | "experiment";
export type ThemeChoice = "light" | "dark" | "auto";
export type OutputName = "steer" | "throttle";
export type CameraMode = "follow" | "fit" | "free";
export type LineMode = "weights" | "flow" | "change" | "mutation";
export type ExperimentTab = "timeline" | "compare" | "training" | "toylab" | "datasets";
export type LiveStatus = "connecting" | "connected" | "disconnected";

export interface NeuronRef {
  layer: number;
  index: number;
}

export interface ConnectionRef {
  layer: number;
  from: number;
  to: number;
}

export interface State {
  /** Where the data comes from; chosen in the top bar, remembered per browser. */
  source: SourceKind;
  /** The live socket's state (live source only). */
  liveStatus: LiveStatus;
  /** The recorded files (recorded source only; also the fallback data of the Experiment tools). */
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

  // the recorded clock: `t` is a fractional index into the recorded frames, `viewKey` the frame
  // looked back at (production: the server's clock and usePlayback.viewStep)
  t: number;
  playing: boolean;
  speed: number;
  viewKey: number | null;

  // selection (production: useSelection; mirrored from it with the live source)
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
  /** The Timeline's run (null: the recorded run, or the newest GA run live). */
  runId: string | null;
  generation: number;
  evoStart: number | null;
}

export interface Actions {
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
  setSpeed: (speed: number) => void;
  /** Look at an earlier step of the selected car (paused only), or null for now. */
  lookAt: (step: number | null) => void;
}

/** The commands the live adapter takes over (production stores and server commands). */
export type Bridge = Pick<
  Actions,
  "selectCar" | "toggleNeuron" | "toggleConnection" | "setWhatIf" | "clearWhatIf" | "play" | "pause" | "stepOnce" | "reset" | "setSpeed" | "lookAt"
>;

let bridge: Bridge | null = null;

/** The live adapter hands the store its commands (null: back to the recorded clock). */
export function attachBridge(next: Bridge | null): void {
  bridge = next;
}

const OUTPUTS: OutputName[] = ["steer", "throttle"];
const SOURCE_KEY = "nd-prototype-source";

function initialView(): ViewId {
  const name = window.location.hash.replace(/^#\/?/, "");
  return name === "network" || name === "experiment" ? name : "environment";
}

/** `?source=live` or `?source=recorded` wins; else the last choice; else recorded (works offline). */
function initialSource(): SourceKind {
  const asked = new URLSearchParams(window.location.search).get("source");
  if (asked === "live" || asked === "recorded") return asked;
  try {
    const saved = window.localStorage.getItem(SOURCE_KEY);
    if (saved === "live" || saved === "recorded") return saved;
  } catch {
    // storage blocked: use the default
  }
  return "recorded";
}

/** Switch the source. The page reloads, so no state of one source leaks into the other. */
export function switchSource(next: SourceKind): void {
  try {
    window.localStorage.setItem(SOURCE_KEY, next);
  } catch {
    // storage blocked: the query string below still carries the choice for this load
  }
  const url = new URL(window.location.href);
  url.searchParams.set("source", next);
  window.location.replace(url.toString());
}

/** The recorded clock's last frame index (0 before the data is loaded). */
function lastKey(data: Recorded | null): number {
  return data === null ? 0 : data.race.frames.step.length - 1;
}

export const useApp = create<State & Actions>()((set, get) => ({
  source: initialSource(),
  liveStatus: "connecting",
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
  runId: null,
  generation: 18,
  evoStart: null,

  setData: (data) =>
    set(
      get().source === "recorded"
        ? { data, car: data.race.scenario.population.champion, generation: data.run.generation, t: 118 }
        : { data },
    ),
  set: (changes) => set(changes),

  selectCar: (index) => {
    if (bridge) return bridge.selectCar(index);
    if (get().car === index) return;
    set({ car: index, whatIf: {}, viewKey: null });
  },

  toggleNeuron: (ref, layers) => {
    if (bridge) return bridge.toggleNeuron(ref, layers);
    const current = get().neuron;
    if (current !== null && current.layer === ref.layer && current.index === ref.index) {
      set({ neuron: null, output: null });
      return;
    }
    const isOutput = ref.layer === layers - 1;
    set({ neuron: ref, connection: null, output: isOutput ? (OUTPUTS[ref.index] ?? null) : null });
  },

  toggleConnection: (ref) => {
    if (bridge) return bridge.toggleConnection(ref);
    const c = get().connection;
    if (c !== null && c.layer === ref.layer && c.from === ref.from && c.to === ref.to) {
      set({ connection: null });
      return;
    }
    set({ connection: ref, neuron: null, output: null });
  },

  setWhatIf: (k, value) => (bridge ? bridge.setWhatIf(k, value) : set({ whatIf: { ...get().whatIf, [k]: value } })),
  clearWhatIf: () => (bridge ? bridge.clearWhatIf() : set({ whatIf: {} })),

  play: () => {
    if (bridge) return bridge.play();
    const last = lastKey(get().data);
    set({ playing: true, viewKey: null, whatIf: {}, t: get().t >= last ? 0 : get().t });
  },
  // Pausing lands on a recorded frame, so every number on screen belongs to one real step.
  pause: () => (bridge ? bridge.pause() : set({ playing: false, t: Math.round(get().t) })),
  stepOnce: () => {
    if (bridge) return bridge.stepOnce();
    set({ playing: false, viewKey: null, whatIf: {}, t: Math.min(lastKey(get().data), Math.floor(get().t) + 1) });
  },
  reset: () => (bridge ? bridge.reset() : set({ playing: false, t: 0, viewKey: null, whatIf: {} })),
  setSpeed: (speed) => (bridge ? bridge.setSpeed(speed) : set({ speed })),
  lookAt: (step) => {
    if (bridge) return bridge.lookAt(step);
    const data = get().data;
    if (get().playing || data === null) return;
    // A step of the recording lives in frame k = step / stepsPerFrame (every 8th step was kept).
    const key = step === null ? null : Math.round(step / data.race.stepsPerFrame);
    set({ viewKey: key === null || key >= Math.round(get().t) ? null : key, whatIf: {} });
  },
}));

// The address hash names the page, as in production (web/src/nav/route.ts).
window.addEventListener("hashchange", () => useApp.setState({ view: initialView() }));

// ── Words and numbers shared by the views ─────────────────────────────────────────────────

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

/** A run id without its time stamp: "20261114-140528-ga-lineage78" → "ga-lineage78". */
export function shortRun(runId: string): string {
  return runId.replace(/^\d{8}-\d{6}-/, "");
}
