import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { FrameMessage, MomentMessage, ScenarioMessage, TrailPoints } from "../api/generated";
import { attachSender, initialLive, updateTrail, useLive } from "./live";
import { activeMoment, usePlayback } from "./playback";
import { useSelection } from "./selection";

// Task 2.11: the trail kept from the frames, the playback clock (which step is looked at) and
// when a look at the past ends.

const points = (steps: number[], replace = false): TrailPoints => ({
  replace,
  steps,
  x: steps.map((s) => s * 10),
  y: steps.map(() => 0),
  action: steps.map((s) => [s / 100, 0.5]),
});

function frame(step: number, car = 0, trail: TrailPoints | null = points([step - 1]), playing = false): FrameMessage {
  return {
    type: "frame",
    step,
    time: step / 30,
    playing,
    speed: 1,
    cars: {
      x: [0, 1],
      y: [0, 0],
      psi: [0, 0],
      v: [0, 0],
      alive: [true, true],
      progress: [0, 0],
      status: [0, 0],
      action: [
        [0, 0],
        [0, 0],
      ],
    },
    selected: { index: car, rays: [1, 2, 3, 4, 5], observation: [0, 0, 0, 0, 0, 0, 0], neuron: null, trace: null, explain: null, trail },
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
    rays: [1, 2, 3, 4, 5],
    observation: [0, 0, 0, 0, 0, 0, 0],
    action: [0.1, 0.5],
    trace: null,
    explain: null,
  };
}

const scenario: ScenarioMessage = {
  type: "scenario",
  track: "train_loop_01",
  policy: "random_networks",
  n_cars: 2,
  seed: null,
  dt: 1 / 30,
  time_limit: 92,
  target_laps: 2,
  v_max: 30,
  wheelbase: 2.5,
  car_half_width: 1,
  ray_angles: [1.05, 0.52, 0, -0.52, -1.05],
  ray_range: 50,
  trail_seconds: 0.2, // 6 steps, so trimming is easy to see
};

let sent: { type: string; step?: number }[];

beforeEach(() => {
  useLive.setState({ ...initialLive });
  useSelection.getState().reset();
  usePlayback.setState({ viewStep: null });
  sent = [];
  attachSender((text) => {
    sent.push(JSON.parse(text));
    return true;
  });
  useLive.getState().receive(scenario);
  useLive.getState().receive(frame(0, 0, points([], true)));
});

afterEach(() => attachSender(null));

describe("updateTrail", () => {
  it("replace starts again, otherwise new points are added", () => {
    let trail = updateTrail(null, 0, points([0, 1, 2], true), 0);
    trail = updateTrail(trail, 0, points([3]), 0);
    expect(trail.steps).toEqual([0, 1, 2, 3]);
    expect(trail.x).toEqual([0, 10, 20, 30]);
    trail = updateTrail(trail, 0, points([7, 8], true), 0);
    expect(trail.steps).toEqual([7, 8]);
  });

  it("drops the steps the server no longer remembers", () => {
    const trail = updateTrail(null, 0, points([0, 1, 2, 3, 4], true), 3);
    expect(trail.steps).toEqual([3, 4]);
    expect(trail.action).toEqual([
      [0.03, 0.5],
      [0.04, 0.5],
    ]);
    expect(updateTrail(trail, 0, points([]), 9).steps).toEqual([]);
  });

  it("points for another car never join the old car's trail", () => {
    const trail = updateTrail(null, 0, points([0, 1], true), 0);
    expect(updateTrail(trail, 1, points([2]), 0)).toEqual({ ...updateTrail(null, 1, points([2]), 0) });
  });
});

describe("the live store keeps the trail from the frames", () => {
  it("adds each frame's points and keeps only trail_seconds of them", () => {
    for (let t = 1; t <= 10; t++) useLive.getState().receive(frame(t));
    // 6 steps remembered at step 10: steps 4 to 9
    expect(useLive.getState().trail?.steps).toEqual([4, 5, 6, 7, 8, 9]);
  });

  it("a new scenario forgets the trail and the moment", () => {
    useLive.getState().receive(frame(1));
    useLive.getState().receive(moment(0));
    useLive.getState().receive(scenario);
    expect(useLive.getState().trail).toBeNull();
    expect(useLive.getState().moment).toBeNull();
  });
});

describe("the playback clock", () => {
  beforeEach(() => {
    for (let t = 1; t <= 5; t++) useLive.getState().receive(frame(t));
  });

  it("view(step) asks the server for that step and clears what-if overrides", () => {
    useSelection.getState().setWhatIf(2, 0.5);
    usePlayback.getState().view(2);
    expect(usePlayback.getState().viewStep).toBe(2);
    expect(sent.at(-1)).toEqual({ type: "inspect_step", step: 2 });
    expect(useSelection.getState().whatIf).toEqual({});
  });

  it("the current step (or later) means now, and nothing is sent", () => {
    usePlayback.getState().view(5);
    expect(usePlayback.getState().viewStep).toBeNull();
    expect(sent.some((c) => c.type === "inspect_step")).toBe(false);
  });

  it("does nothing while playing", () => {
    useLive.getState().receive(frame(6, 0, points([5]), true));
    usePlayback.getState().view(2);
    expect(usePlayback.getState().viewStep).toBeNull();
  });

  it("the moment shows only when it answers the step looked at, for the car on screen", () => {
    usePlayback.getState().view(2);
    expect(activeMoment(useLive.getState(), 2)).toBeNull(); // not answered yet
    useLive.getState().receive(moment(3));
    expect(activeMoment(useLive.getState(), 2)).toBeNull(); // an answer about another step
    useLive.getState().receive(moment(2));
    expect(activeMoment(useLive.getState(), 2)?.step).toBe(2);
    expect(activeMoment(useLive.getState(), null)).toBeNull();
  });

  it("stepping on, choosing another car or loading a scenario goes back to now", () => {
    usePlayback.getState().view(2);
    useLive.getState().receive(moment(2));
    useLive.getState().receive(frame(5, 0, points([]))); // a frame at the same step (e.g. pause)
    expect(usePlayback.getState().viewStep).toBe(2);
    expect(useLive.getState().moment).not.toBeNull();
    useLive.getState().receive(frame(6));
    expect(usePlayback.getState().viewStep).toBeNull();
    expect(useLive.getState().moment).toBeNull();

    usePlayback.getState().view(3);
    useLive.getState().receive(frame(6, 1, points([0, 1, 2], true)));
    expect(usePlayback.getState().viewStep).toBeNull();

    usePlayback.getState().view(1);
    useLive.getState().receive(scenario);
    expect(usePlayback.getState().viewStep).toBeNull();
  });

  it("backToLive returns to now", () => {
    usePlayback.getState().view(2);
    usePlayback.getState().backToLive();
    expect(usePlayback.getState().viewStep).toBeNull();
  });
});
