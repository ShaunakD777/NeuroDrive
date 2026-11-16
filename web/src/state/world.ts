import { create } from "zustand";

import type {
  EpisodeEndMessage,
  ExplanationMessage,
  Explanation,
  FrameMessage,
  GhostPaths,
  MomentMessage,
  NetworkMessage,
  NetworkTrace,
  NeuronReadout,
  ScenarioMessage,
  TrackGeometry,
} from "../api/generated";
import { type Trail, useLive } from "./live";
import { activeMoment, usePlayback } from "./playback";
import { useSelection } from "./selection";

// The world selector (task S2, ADR-063 D2): one answer to "what is on screen for the selected car
// right now", built from the three stores (live, playback, selection). Every rule that picks
// between the live frame and a look back, or decides whether a what-if answer still applies, is
// here once (docs/ARCHITECTURE.md §8.1, §8.2). Nothing here computes driving or network values:
// it only chooses among the server's messages and reshapes them.

/** One car at one moment: one row of the frame's columns (`FrameMessage.cars`). */
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
 * The selected car at the moment shown: now (`frame.selected`) or the step looked at (the
 * server's `moment` answer). What it senses and, if a network drives it, every value of its
 * forward pass and Python's explanation of it.
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
  /** The neuron driver's read-out now; null for other drivers and while looking back. */
  neuron: NeuronReadout | null;
}

/** A detail whose car is driven by a network: the Network view needs both. */
export type NetworkDetail = CarDetail & { trace: NetworkTrace; explain: Explanation };

export function hasNetwork(detail: CarDetail | null): detail is NetworkDetail {
  return detail !== null && detail.trace !== null && detail.explain !== null;
}

/** Python's answer to a what-if (`ExplanationMessage`), reduced to what the views draw. */
export interface WhatIfView {
  /** The network's outputs for the car's real inputs. */
  base: number[];
  /** The output layer's activations for the changed inputs. */
  tried: number[];
  /** The outputs for the changed inputs, as the server computed them. */
  predicted: number[];
  /** Where the car would go in a few seconds holding each action (physics in Python). */
  ghost: GhostPaths;
}

export interface World {
  track: TrackGeometry;
  scenario: ScenarioMessage;
  step: number;
  time: number;
  playing: boolean;
  speed: number;
  /** The episode has ended (every car stopped, or the time limit). */
  ended: boolean;
  cars: CarPose[];
  /** The selected car, as the server last confirmed it. */
  selected: number;
  /** The earlier step being shown, or null for now. */
  lookStep: number | null;
  /** The selected car at the shown moment; null while the server has not sent it yet. */
  shown: CarDetail | null;
  /** The selected car's decision trail; null when the stored trail is another car's. */
  trail: Trail | null;
  /** The what-if answer, only for the selected car at the frame's step while inputs are changed. */
  whatIf: WhatIfView | null;
  /** The selected car's network weights; null until they are this car's. */
  network: NetworkMessage | null;
  /** Each car's fitness f on the raced start once the race has ended (GA race only). */
  endFitness: number[] | null;
}

/** What the world is built from: the parts of the three stores it reads. */
export interface WorldInputs {
  live: {
    scenario: ScenarioMessage | null;
    geometry: TrackGeometry | null;
    frame: FrameMessage | null;
    network: NetworkMessage | null;
    whatIfResult: ExplanationMessage | null;
    trail: Trail | null;
    moment: MomentMessage | null;
    episodeEnd: EpisodeEndMessage | null;
  };
  /** The playback clock's step looked at (null = now). */
  viewStep: number | null;
  /** The selection's changed inputs (index → value). */
  overrides: Record<number, number>;
}

const posesCache = new WeakMap<FrameMessage["cars"], CarPose[]>();

/** The frame's columns as one pose per car; the same frame always gives the same list. */
export function posesOf(cars: FrameMessage["cars"]): CarPose[] {
  let poses = posesCache.get(cars);
  if (poses === undefined) {
    poses = cars.x.map((x, i) => ({
      x,
      y: cars.y[i],
      psi: cars.psi[i],
      v: cars.v[i],
      progress: cars.progress[i],
      steer: cars.action[i]?.[0] ?? 0,
      throttle: cars.action[i]?.[1] ?? 0,
      status: cars.status[i],
    }));
    posesCache.set(cars, poses);
  }
  return poses;
}

function momentDetail(moment: MomentMessage): CarDetail {
  return {
    index: moment.car,
    step: moment.step,
    pose: {
      x: moment.x,
      y: moment.y,
      psi: moment.psi,
      v: moment.v,
      progress: moment.progress,
      steer: moment.action[0] ?? 0,
      throttle: moment.action[1] ?? 0,
      status: moment.status,
    },
    rays: moment.rays,
    observation: moment.observation,
    action: moment.action,
    trace: moment.trace,
    explain: moment.explain,
    neuron: null,
  };
}

/**
 * The world on screen, or null until the server has sent a scenario, its track and a frame (or
 * while the stored track is not the scenario's, between a track change and its geometry).
 *
 * The rules (ARCHITECTURE.md §8.2):
 * 1. The look-back moment replaces the present only when a step is looked at and the moment is
 *    about that step of the car on screen (`activeMoment`).
 * 2. A what-if answer applies only while at least one input is changed, and only to the car and
 *    step it was asked about.
 * 3. The trail and the network weights are shown only when they are the selected car's.
 */
export function buildWorld({ live, viewStep, overrides }: WorldInputs): World | null {
  const { scenario, geometry, frame } = live;
  if (scenario === null || geometry === null || frame === null || geometry.name !== scenario.track) return null;
  const cars = posesOf(frame.cars);
  const now = frame.selected;
  const selected = now?.index ?? 0;
  const moment = activeMoment(live, viewStep);

  let shown: CarDetail | null = null;
  if (moment !== null) {
    shown = momentDetail(moment);
  } else if (now) {
    shown = {
      index: now.index,
      step: frame.step,
      pose: cars[now.index],
      rays: now.rays,
      observation: now.observation,
      action: frame.cars.action[now.index] ?? [0, 0],
      trace: now.trace,
      explain: now.explain,
      neuron: now.neuron,
    };
  }

  const result = live.whatIfResult;
  const asking = Object.keys(overrides).length > 0;
  const whatIf: WhatIfView | null =
    asking && result !== null && now !== null && result.car === now.index && result.step === frame.step
      ? {
          base: result.base_outputs,
          tried: result.trace.activations[result.trace.activations.length - 1] ?? [],
          predicted: result.predicted_outputs,
          ghost: result.ghost,
        }
      : null;

  return {
    track: geometry,
    scenario,
    step: frame.step,
    time: frame.time,
    playing: frame.playing,
    speed: frame.speed,
    ended: live.episodeEnd !== null,
    cars,
    selected,
    lookStep: moment === null ? null : moment.step,
    shown,
    trail: live.trail !== null && live.trail.car === selected ? live.trail : null,
    whatIf,
    network: live.network !== null && live.network.car === selected ? live.network : null,
    endFitness: live.episodeEnd?.fitness ?? null,
  };
}

// ── The derived store ────────────────────────────────────────────────────────────────────

const useWorldStore = create<{ world: World | null }>(() => ({ world: null }));

let previousInputs: readonly unknown[] = [];

/** Rebuild the world only when one of its inputs is a different object or value than last time. */
function recompute(): void {
  const live = useLive.getState();
  const viewStep = usePlayback.getState().viewStep;
  const overrides = useSelection.getState().whatIf;
  const inputs = [
    live.scenario,
    live.geometry,
    live.frame,
    live.network,
    live.whatIfResult,
    live.trail,
    live.moment,
    live.episodeEnd,
    viewStep,
    overrides,
  ];
  const same = inputs.length === previousInputs.length && inputs.every((v, i) => Object.is(v, previousInputs[i]));
  if (same) return;
  previousInputs = inputs;
  useWorldStore.setState({ world: buildWorld({ live, viewStep, overrides }) });
}

// Follow the three stores for the whole life of the page (as playback.ts follows the live store).
useLive.subscribe(recompute);
usePlayback.subscribe(recompute);
useSelection.subscribe(recompute);
recompute();

/** The world on screen, for React components (null until the server has sent a track and a frame). */
export function useWorld(): World | null {
  return useWorldStore((s) => s.world);
}

/** The world on screen, for code outside React (the canvas loop, event handlers). */
export function getWorld(): World | null {
  return useWorldStore.getState().world;
}
