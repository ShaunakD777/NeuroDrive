// The world snapshot: everything the Environment and Network views draw, in one shape, whichever
// source filled it. Two adapters fill it:
//   - recorded.ts, from the JSON files in public/recorded/ (no server; its few stand-in sums are
//     in src/mock/mockServer.ts and nowhere else);
//   - live.ts, from the production app's own stores (web/src/state/*), i.e. from the messages
//     the Python server sends.
// The views read only this. Where a type has the same shape as a production message it is that
// message's type, imported from web/src/api/generated.ts, so a view written against the snapshot
// is already written against the server's contract.

import type {
  CheckpointDiff,
  Explanation,
  MutationView,
  NetworkMessage,
  NetworkTrace,
  NeuronReadout,
  PopulationInfo,
  ScenarioMessage,
  TrackGeometry,
} from "@prod/api/generated";

export type SourceKind = "recorded" | "live";

export type Track = TrackGeometry;
export type Population = PopulationInfo;
export type Network = Omit<NetworkMessage, "type">;
export type Mutation = MutationView;
export type Diff = CheckpointDiff;
export type Policy = ScenarioMessage["policy"];

/** What is on the track (`ScenarioMessage` without its type tag). */
export type Scenario = Omit<ScenarioMessage, "type">;

/** Car status codes, as the server sends them (`CarColumns.status`). */
export const STATUS_CODE = { running: 0, crashed: 1, finished: 2, stalled: 3, timeout: 4 } as const;

/** One car at one moment. Columns of `FrameMessage.cars`, one row. */
export interface CarPose {
  x: number;
  y: number;
  psi: number;
  v: number;
  progress: number;
  steer: number;
  throttle: number;
  status: number;
}

/**
 * The selected car at the moment shown (now, or the step looked back at): what it senses and, if
 * it is driven by a network, every value of the forward pass and Python's explanation of it.
 * Live, this is `frame.selected` (now) or the `moment` answer (looking back).
 */
export interface CarDetail {
  index: number;
  step: number;
  pose: CarPose;
  rays: number[];
  observation: number[];
  action: number[];
  /** null for drivers without a multilayer network (keyboard, constant, neuron, expert). */
  trace: NetworkTrace | null;
  explain: Explanation | null;
  /** The neuron driver's read-out, null for every other driver. */
  neuron: NeuronReadout | null;
}

/** A detail whose car is driven by a network: the Network view needs all three. */
export type NetworkDetail = CarDetail & { trace: NetworkTrace; explain: Explanation };

export function hasNetwork(detail: CarDetail | null): detail is NetworkDetail {
  return detail !== null && detail.trace !== null && detail.explain !== null;
}

/** The selected car's remembered drive: at `steps[i]` it was at (`x[i]`, `y[i]`) and chose `action[i]`. */
export interface TrailView {
  steps: number[];
  x: number[];
  y: number[];
  action: number[][];
}

/** Python's answer to a what-if (`ExplanationMessage`), reduced to what the views draw. */
export interface WhatIfView {
  base: number[];
  tried: number[];
  predicted: number[];
  /** Where the car would go in 2 s holding each action; from the physics in Python. */
  ghost: { real: number[][]; whatIf: number[][]; end: [number, number, number] } | null;
}

/** The selected network's lineage or checkpoint change, with a reason when there is none. */
export interface Origin {
  mutation: Mutation | null;
  /** In a checkpoint replay: what changed since the previous checkpoint. */
  diff: Diff | null;
  diffFrom: number | null;
  /** Why there is no mutation (generation 1, the ghost, a run trained before task 7.8). */
  note: string | null;
}

export interface World {
  source: SourceKind;
  track: Track;
  scenario: Scenario;
  step: number;
  time: number;
  playing: boolean;
  speed: number;
  /** The episode has ended (every car stopped, or the time limit). */
  ended: boolean;
  cars: CarPose[];
  /** The selected car, as the source last confirmed it. */
  selected: number;
  /** The earlier step being looked at, null for now. */
  lookStep: number | null;
  /** The selected car at the shown moment; null while the source has not sent it yet. */
  shown: CarDetail | null;
  trail: TrailView | null;
  whatIf: WhatIfView | null;
  network: Network | null;
  origin: Origin;
  /**
   * c[layer][to][from] = weight × source value for every connection, at the shown moment. Only
   * the recorded source has it (computed by the stand-in); the server sends contributions for
   * the output layer only (`explain.receipts`), so live it is null (PRODUCTION_MAPPING.md §6.1).
   */
  flow: number[][][] | null;
  /** Each car's fitness f on the raced start, once the race has ended (population driver only). */
  endFitness: number[] | null;
}

/** What a car is in its scenario: a network of a GA generation (with its role), the ghost, or just a car. */
export type CarRole = "champion" | "elite" | "child" | "ghost" | "car";

export function carRole(scenario: Scenario, i: number): CarRole {
  const info = scenario.population;
  if (info === null) return "car";
  if (i === info.ghost_car) return "ghost";
  if (i === info.champion) return "champion";
  if (info.elite.includes(i)) return "elite";
  return "child";
}

export function carName(scenario: Scenario, i: number): string {
  const info = scenario.population;
  if (info !== null && i === info.ghost_car) return `Ghost · gen ${info.ghost_unit} champion`;
  return `Car ${i + 1}`;
}

/** Drivers whose cars are driven by a multilayer network the Network view can draw. */
export function drivesNetworks(policy: Policy): boolean {
  return policy === "random_networks" || policy === "checkpoint" || policy === "population";
}

export const POLICY_LABEL: Record<Policy, string> = {
  keyboard: "Keyboard (you drive)",
  constant: "Constant (steer 0, throttle 0.3)",
  neuron: "Neuron (you set its weights)",
  random_networks: "Random networks (one per car)",
  expert: "Expert (scripted, reads the track)",
  checkpoint: "Checkpoint replay",
  population: "GA generation",
};
