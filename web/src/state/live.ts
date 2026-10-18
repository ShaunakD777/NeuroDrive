import { create } from "zustand";

import type {
  ActivationCurves,
  EpisodeEndMessage,
  ErrorMessage,
  FrameMessage,
  KeyState,
  LoadScenario,
  NeuronReadout,
  Pause,
  Play,
  Reset,
  ScenarioMessage,
  SelectCar,
  SetParam,
  SetSpeed,
  Step,
  TrackGeometry,
  TrackInfo,
} from "../api/generated";
import type { ConnectionStatus } from "../api/ws";

// The live store: everything the browser knows about the running simulation. The Python server
// owns the simulation; this only remembers what it last said (docs/ARCHITECTURE.md §7) and sends
// commands back. Nothing here computes driving.

/** Every command the browser can send over /ws/live. */
export type Command =
  | LoadScenario
  | Play
  | Pause
  | Step
  | Reset
  | SetSpeed
  | SelectCar
  | KeyState
  | SetParam;

/** The messages the server sends that the store keeps (hello only confirms the connection). */
export type ServerMessage = ScenarioMessage | FrameMessage | EpisodeEndMessage | ErrorMessage;

export interface DriveInput {
  steer: number;
  throttle: number;
}

/** What a `load_scenario` message asks for. */
export interface ScenarioRequest {
  track: string;
  policy: "keyboard" | "constant" | "neuron";
  n_cars: number;
  /** null: every car starts at the track's start line; a number: a seeded random start. */
  seed: number | null;
}

/** The neuron driver's settings, as the server last reported them. */
export interface NeuronSettings {
  w_left: number;
  w_right: number;
  bias: number;
  throttle: number;
  activation: "tanh" | "sigmoid" | "relu" | "linear";
}

export interface LiveState {
  status: ConnectionStatus;
  /** True once the socket has been connected at least once (so a later "connected" is a reconnect). */
  everConnected: boolean;
  /**
   * The neuron settings last seen. The server keeps them for a whole session (through Reset and
   * scenario changes); a reconnect starts a new session, so the browser sends these back.
   */
  neuronSettings: NeuronSettings | null;
  /** Set on a reconnect: restore `neuronSettings` when the first neuron readout arrives. */
  restoreNeuron: boolean;
  tracks: TrackInfo[];
  /** The scenario the user last asked for; loaded again after a reconnect. */
  wanted: ScenarioRequest | null;
  scenario: ScenarioMessage | null;
  geometry: TrackGeometry | null;
  /** The activation functions sampled by the server, for drawing curves (null until fetched). */
  activations: ActivationCurves | null;
  frame: FrameMessage | null;
  episodeEnd: EpisodeEndMessage | null;
  /** The last error the server reported, or null. */
  error: string | null;
  /** The steer and throttle last sent as `key_state`. */
  keys: DriveInput;
}

export interface LiveActions {
  setStatus: (status: ConnectionStatus) => void;
  setTracks: (tracks: TrackInfo[]) => void;
  setGeometry: (geometry: TrackGeometry) => void;
  setActivations: (activations: ActivationCurves) => void;
  /**
   * Ask the server to load a scenario. `changes` are applied on top of the scenario already
   * loaded (or the one last asked for), so changing one setting keeps the others. Remembered so
   * a reconnect loads the same scenario again.
   */
  loadScenario: (changes: Partial<ScenarioRequest>) => void;
  /** File one server message into the store. */
  receive: (message: unknown) => void;
  /** Send held-key input to the server if it differs from what was last sent. */
  setKeys: (input: DriveInput) => void;
  /** Send any command to the server. Returns false if the socket is not open. */
  send: (command: Command) => boolean;
}

type Sender = (text: string) => boolean;
let sender: Sender | null = null;

/** The app gives the store the function that writes to the socket (null when there is none). */
export function attachSender(fn: Sender | null): void {
  sender = fn;
}

const NO_KEYS: DriveInput = { steer: 0, throttle: 0 };

export const initialLive: LiveState = {
  status: "connecting",
  everConnected: false,
  neuronSettings: null,
  restoreNeuron: false,
  tracks: [],
  wanted: null,
  scenario: null,
  geometry: null,
  activations: null,
  frame: null,
  episodeEnd: null,
  error: null,
  keys: NO_KEYS,
};

const PARAM_ORDER = ["w_left", "w_right", "bias", "throttle", "activation"] as const;

/** The settings part of a neuron readout. */
export function settingsOf(readout: NeuronReadout): NeuronSettings {
  return {
    w_left: readout.weights[0],
    w_right: readout.weights[1],
    bias: readout.bias,
    throttle: readout.throttle,
    activation: readout.activation,
  };
}

function isServerMessage(value: unknown): value is ServerMessage {
  if (typeof value !== "object" || value === null) return false;
  const type = (value as { type?: unknown }).type;
  return type === "scenario" || type === "frame" || type === "episode_end" || type === "error";
}

export const useLive = create<LiveState & LiveActions>()((set, get) => ({
  ...initialLive,

  setStatus: (status) => {
    if (status !== "connected") return set({ status });
    // A second "connected" means the server was lost and found again: a fresh session.
    const reconnect = get().everConnected;
    set({ status, everConnected: true, restoreNeuron: reconnect && get().neuronSettings !== null });
  },
  setTracks: (tracks) => set({ tracks }),
  setGeometry: (geometry) => set({ geometry }),
  setActivations: (activations) => set({ activations }),

  loadScenario: (changes) => {
    const { scenario, wanted, tracks } = get();
    // Build on what was last asked for (it may not have been answered yet), else on what is loaded.
    const base: Partial<ScenarioRequest> =
      wanted ??
      (scenario === null
        ? {}
        : {
            track: scenario.track,
            policy: scenario.policy,
            n_cars: scenario.n_cars,
            seed: scenario.seed,
          });
    const track = changes.track ?? base.track ?? tracks[0]?.name;
    if (track === undefined) return; // nothing to load yet
    const request: ScenarioRequest = {
      track,
      policy: changes.policy ?? base.policy ?? "keyboard",
      n_cars: changes.n_cars ?? base.n_cars ?? 1,
      seed: changes.seed !== undefined ? changes.seed : (base.seed ?? null),
    };
    set({ wanted: request });
    get().send({ type: "load_scenario", ...request });
  },

  receive: (message) => {
    if (!isServerMessage(message)) return;
    switch (message.type) {
      case "scenario":
        // A new scenario: forget the old frame and ending. The server starts a new keyboard
        // driver with no keys held, so the keys we last sent no longer apply either.
        set({ scenario: message, frame: null, episodeEnd: null, error: null, keys: NO_KEYS });
        break;
      case "frame": {
        // A frame at step 0 means a fresh start (after load or reset): any old ending is stale.
        set(message.step === 0 ? { frame: message, episodeEnd: null } : { frame: message });
        const readout = message.selected?.neuron;
        if (readout) {
          const seen = settingsOf(readout);
          const saved = get().neuronSettings;
          if (get().restoreNeuron && saved !== null) {
            set({ restoreNeuron: false });
            for (const name of PARAM_ORDER) {
              if (saved[name] !== seen[name]) get().send({ type: "set_param", name, value: saved[name] });
            }
          } else {
            set({ neuronSettings: seen });
          }
        }
        break;
      }
      case "episode_end":
        set({ episodeEnd: message });
        break;
      case "error":
        set({ error: message.message });
        break;
    }
  },

  setKeys: (input) => {
    const last = get().keys;
    if (last.steer === input.steer && last.throttle === input.throttle) return;
    if (get().send({ type: "key_state", steer: input.steer, throttle: input.throttle })) {
      set({ keys: input });
    }
  },

  send: (command) => (sender === null ? false : sender(JSON.stringify(command))),
}));
