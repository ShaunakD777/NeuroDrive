import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type {
  EpisodeEndMessage,
  ExplanationMessage,
  FrameMessage,
  MomentMessage,
  NetworkMessage,
  ScenarioMessage,
  TrackGeometry,
} from "../api/generated";
import { attachSender, initialLive, type Trail, useLive } from "./live";
import { usePlayback } from "./playback";
import { useSelection } from "./selection";
import { buildWorld, getWorld, hasNetwork, posesOf, type WorldInputs } from "./world";

// Task S2: the world selector. The cases of playback.test.ts (which moment is shown) and of
// inspect.test.ts (when a what-if answer applies), asked of the one selector.

const scenario: ScenarioMessage = {
  type: "scenario",
  track: "train_loop_01",
  policy: "random_networks",
  n_cars: 2,
  seed: null,
  checkpoint: null,
  start_shift: null,
  population: null,
  dt: 1 / 30,
  time_limit: 92,
  target_laps: 2,
  v_max: 30,
  wheelbase: 2.5,
  car_half_width: 1,
  ray_angles: [1.05, 0.52, 0, -0.52, -1.05],
  ray_range: 50,
  trail_seconds: 0.2,
};

const geometry: TrackGeometry = {
  name: "train_loop_01",
  centerline: [],
  left: [],
  right: [],
  length: 100,
  split: "train",
  start_s0: 0,
  width: 10,
};

function frame(step: number, car = 0, playing = false): FrameMessage {
  return {
    type: "frame",
    step,
    time: step / 30,
    playing,
    speed: 1,
    cars: {
      x: [step, 100 + step],
      y: [0, 5],
      psi: [0, 0.5],
      v: [3, 4],
      alive: [true, true],
      progress: [0.1, 0.2],
      status: [0, 1],
      action: [
        [0.1, 0.5],
        [-0.2, 0.7],
      ],
    },
    selected: {
      index: car,
      rays: [1, 2, 3, 4, 5],
      observation: [0, 0, 0, 0, 0, 0, 0],
      neuron: null,
      trace: null,
      explain: null,
      trail: { replace: false, steps: [], x: [], y: [], action: [] },
    },
  };
}

function moment(step: number, car = 0): MomentMessage {
  return {
    type: "moment",
    car,
    step,
    time: step / 30,
    x: step * 10,
    y: 0,
    psi: 0,
    v: 1,
    progress: 0,
    status: 0,
    rays: [9, 9, 9, 9, 9],
    observation: [1, 1, 1, 1, 1, 1, 1],
    action: [0.3, 0.6],
    trace: null,
    explain: null,
  };
}

const ghost = {
  seconds: 2,
  real_action: [0.1, 0.5],
  what_if_action: [0.4, 0.5],
  real: [
    [0, 0],
    [1, 0],
  ],
  what_if: [
    [0, 0],
    [1, 0.2],
  ],
  real_end: [1, 0, 0],
  what_if_end: [1, 0.2, 0.1],
};

function whatIfResult(car: number, step: number): ExplanationMessage {
  return {
    type: "explanation",
    car,
    step,
    base_outputs: [0.1, 0.5],
    predicted_outputs: [0.4, 0.5],
    trace: { activations: [[0], [0.3], [0.4, 0.5]] },
    ghost,
  } as unknown as ExplanationMessage;
}

const network = (car: number) => ({ type: "network", car, layers: [7, 2], weights: [], biases: [] }) as unknown as NetworkMessage;
const trail = (car: number): Trail => ({ car, steps: [1], x: [1], y: [0], action: [[0, 0]] });

function inputs(changes: Partial<WorldInputs["live"]> = {}, viewStep: number | null = null, overrides: Record<number, number> = {}): WorldInputs {
  return {
    live: {
      scenario,
      geometry,
      frame: frame(6),
      network: null,
      whatIfResult: null,
      trail: null,
      moment: null,
      episodeEnd: null,
      ...changes,
    },
    viewStep,
    overrides,
  };
}

describe("buildWorld: when there is a world", () => {
  it("is null until the scenario, its track and a frame are there", () => {
    expect(buildWorld(inputs({ scenario: null }))).toBeNull();
    expect(buildWorld(inputs({ geometry: null }))).toBeNull();
    expect(buildWorld(inputs({ frame: null }))).toBeNull();
    expect(buildWorld(inputs())).not.toBeNull();
  });

  it("is null while the stored track is not the scenario's (between a track change and its geometry)", () => {
    expect(buildWorld(inputs({ geometry: { ...geometry, name: "monza" } }))).toBeNull();
  });
});

describe("buildWorld: the present", () => {
  it("turns the frame's columns into one pose per car", () => {
    const world = buildWorld(inputs())!;
    expect(world.cars).toHaveLength(2);
    expect(world.cars[1]).toEqual({ x: 106, y: 5, psi: 0.5, v: 4, progress: 0.2, steer: -0.2, throttle: 0.7, status: 1 });
    expect(world.step).toBe(6);
    expect(world.playing).toBe(false);
  });

  it("gives the same pose list for the same frame, so a rebuild does not redraw every car", () => {
    const f = frame(6);
    expect(posesOf(f.cars)).toBe(posesOf(f.cars));
  });

  it("shows the selected car now: the frame's rays and inputs, its pose and action from the columns", () => {
    const world = buildWorld(inputs({ frame: frame(6, 1) }))!;
    expect(world.selected).toBe(1);
    expect(world.lookStep).toBeNull();
    expect(world.shown?.index).toBe(1);
    expect(world.shown?.step).toBe(6);
    expect(world.shown?.pose.x).toBe(106);
    expect(world.shown?.action).toEqual([-0.2, 0.7]);
    expect(world.shown?.rays).toEqual([1, 2, 3, 4, 5]);
  });

  it("has no shown car while the frame names none", () => {
    const f = { ...frame(6), selected: null } as unknown as FrameMessage;
    const world = buildWorld(inputs({ frame: f }))!;
    expect(world.selected).toBe(0);
    expect(world.shown).toBeNull();
  });

  it("knows whether the shown car is driven by a network", () => {
    expect(hasNetwork(buildWorld(inputs())!.shown)).toBe(false);
    expect(hasNetwork(null)).toBe(false);
  });
});

describe("buildWorld: a look back (the cases of playback.test.ts)", () => {
  it("shows the moment only when it answers the step looked at, for the car on screen", () => {
    // Looking at step 2, no answer yet: still the present.
    expect(buildWorld(inputs({}, 2))!.lookStep).toBeNull();
    // An answer about another step: still the present.
    expect(buildWorld(inputs({ moment: moment(3) }, 2))!.shown?.step).toBe(6);
    // The answer about step 2: the moment, with its own pose, rays and action.
    const world = buildWorld(inputs({ moment: moment(2) }, 2))!;
    expect(world.lookStep).toBe(2);
    expect(world.shown?.step).toBe(2);
    expect(world.shown?.pose.x).toBe(20);
    expect(world.shown?.pose.steer).toBe(0.3);
    expect(world.shown?.rays).toEqual([9, 9, 9, 9, 9]);
    expect(world.shown?.neuron).toBeNull();
    // Back to now: the present, although the answer is still stored.
    expect(buildWorld(inputs({ moment: moment(2) }, null))!.shown?.step).toBe(6);
  });

  it("ignores an answer about another car", () => {
    const world = buildWorld(inputs({ frame: frame(6, 1), moment: moment(2, 0) }, 2))!;
    expect(world.lookStep).toBeNull();
    expect(world.shown?.index).toBe(1);
  });

  it("keeps every car's pose at the present while one car is looked back at", () => {
    const world = buildWorld(inputs({ moment: moment(2) }, 2))!;
    expect(world.step).toBe(6);
    expect(world.cars[0].x).toBe(6);
  });
});

describe("buildWorld: the what-if answer (the cases of inspect.test.ts)", () => {
  const at = (car: number, step: number, overrides: Record<number, number>) =>
    buildWorld(inputs({ frame: frame(step, car), whatIfResult: whatIfResult(0, 12) }, null, overrides))!.whatIf;

  it("applies to the car and step it was asked about, while an input is changed", () => {
    const view = at(0, 12, { 0: 0.5 });
    expect(view).not.toBeNull();
    expect(view?.base).toEqual([0.1, 0.5]);
    expect(view?.tried).toEqual([0.4, 0.5]); // the output layer's activations
    expect(view?.predicted).toEqual([0.4, 0.5]);
    expect(view?.ghost).toBe(ghost); // the server's own shape
  });

  it("is dropped with no changed input, at another step, for another car, or with no answer", () => {
    expect(at(0, 12, {})).toBeNull();
    expect(at(0, 13, { 0: 0.5 })).toBeNull();
    expect(at(1, 12, { 0: 0.5 })).toBeNull();
    expect(buildWorld(inputs({ frame: frame(12, 0) }, null, { 0: 0.5 }))!.whatIf).toBeNull();
  });
});

describe("buildWorld: the selected car's trail, weights and the end of the race", () => {
  it("keeps the trail and the network only when they are the selected car's", () => {
    const mine = buildWorld(inputs({ frame: frame(6, 1), trail: trail(1), network: network(1) }))!;
    expect(mine.trail?.car).toBe(1);
    expect(mine.network?.car).toBe(1);
    const theirs = buildWorld(inputs({ frame: frame(6, 1), trail: trail(0), network: network(0) }))!;
    expect(theirs.trail).toBeNull();
    expect(theirs.network).toBeNull();
  });

  it("knows when the episode ended and, for a GA race, each car's fitness", () => {
    expect(buildWorld(inputs())!.ended).toBe(false);
    const end = { type: "episode_end", fitness: [12.5, 3], laps: [], progress: [], status: [], step: 6, time: 0.2 } as EpisodeEndMessage;
    const world = buildWorld(inputs({ episodeEnd: end }))!;
    expect(world.ended).toBe(true);
    expect(world.endFitness).toEqual([12.5, 3]);
    expect(buildWorld(inputs({ episodeEnd: { ...end, fitness: null } }))!.endFitness).toBeNull();
  });
});

describe("the world store follows the three stores", () => {
  beforeEach(() => {
    useLive.setState({ ...initialLive });
    useSelection.getState().reset();
    usePlayback.setState({ viewStep: null });
    attachSender(() => true);
  });

  afterEach(() => attachSender(null));

  it("is null until the server has sent a scenario, its track and a frame, then follows the frames", () => {
    expect(getWorld()).toBeNull();
    useLive.getState().receive(scenario);
    useLive.getState().setGeometry(geometry);
    expect(getWorld()).toBeNull();
    useLive.getState().receive(frame(0));
    expect(getWorld()?.step).toBe(0);
    useLive.getState().receive(frame(1));
    expect(getWorld()?.step).toBe(1);
  });

  it("follows a look back through the playback clock and drops it when the simulation moves on", () => {
    useLive.getState().receive(scenario);
    useLive.getState().setGeometry(geometry);
    useLive.getState().receive(frame(6));
    usePlayback.getState().view(2);
    expect(getWorld()?.lookStep).toBeNull(); // asked, not answered yet
    useLive.getState().receive(moment(2));
    expect(getWorld()?.lookStep).toBe(2);
    useLive.getState().receive(frame(7));
    expect(getWorld()?.lookStep).toBeNull();
    expect(getWorld()?.shown?.step).toBe(7);
  });

  it("does not rebuild when nothing it reads changed", () => {
    useLive.getState().receive(scenario);
    useLive.getState().setGeometry(geometry);
    useLive.getState().receive(frame(6));
    const before = getWorld();
    useLive.setState({ error: "unrelated" }); // the live store changed, but not a world input
    useSelection.setState({ neuron: { layer: 1, index: 0 } });
    expect(getWorld()).toBe(before);
    useSelection.getState().setWhatIf(0, 0.5); // a changed input is a world input
    expect(getWorld()).not.toBe(before);
  });
});
