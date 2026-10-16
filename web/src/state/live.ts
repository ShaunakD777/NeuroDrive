import { create } from "zustand";

import type {
  EpisodeEndMessage,
  ErrorMessage,
  FrameMessage,
  KeyState,
  LoadScenario,
  Pause,
  Play,
  Reset,
  ScenarioMessage,
  SelectCar,
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
  | KeyState;

/** The messages the server sends that the store keeps (hello only confirms the connection). */
export type ServerMessage = ScenarioMessage | FrameMessage | EpisodeEndMessage | ErrorMessage;

export interface DriveInput {
  steer: number;
  throttle: number;
}

export interface LiveState {
  status: ConnectionStatus;
  tracks: TrackInfo[];
  /** The track the user wants loaded; used again after a reconnect. */
  wantedTrack: string | null;
  scenario: ScenarioMessage | null;
  geometry: TrackGeometry | null;
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
  /** Remember which track to load, and tell the server to load it. */
  loadTrack: (name: string) => void;
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
  tracks: [],
  wantedTrack: null,
  scenario: null,
  geometry: null,
  frame: null,
  episodeEnd: null,
  error: null,
  keys: NO_KEYS,
};

function isServerMessage(value: unknown): value is ServerMessage {
  if (typeof value !== "object" || value === null) return false;
  const type = (value as { type?: unknown }).type;
  return type === "scenario" || type === "frame" || type === "episode_end" || type === "error";
}

export const useLive = create<LiveState & LiveActions>()((set, get) => ({
  ...initialLive,

  setStatus: (status) => set({ status }),
  setTracks: (tracks) => set({ tracks }),
  setGeometry: (geometry) => set({ geometry }),

  loadTrack: (name) => {
    set({ wantedTrack: name });
    get().send({ type: "load_scenario", track: name, policy: "keyboard", n_cars: 1, seed: null });
  },

  receive: (message) => {
    if (!isServerMessage(message)) return;
    switch (message.type) {
      case "scenario":
        // A new scenario: forget the old frame and ending. The server starts a new keyboard
        // driver with no keys held, so the keys we last sent no longer apply either.
        set({ scenario: message, frame: null, episodeEnd: null, error: null, keys: NO_KEYS });
        break;
      case "frame":
        // A frame at step 0 means a fresh start (after load or reset): any old ending is stale.
        set(message.step === 0 ? { frame: message, episodeEnd: null } : { frame: message });
        break;
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
