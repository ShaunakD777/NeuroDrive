import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { FrameMessage, ScenarioMessage, SelectedCar } from "../api/generated";
import { attachSender, initialLive, useLive } from "./live";
import { raysFor, useSelection } from "./selection";

// Selection linking (task 2.8): the store transitions, the select_car command, following the
// server's frames, and which rays a selection highlights.

const LAYERS = 4; // inputs, hidden 1, hidden 2, outputs

function frame(selectedIndex: number, step = 1): FrameMessage {
  return {
    type: "frame",
    step,
    time: step / 30,
    playing: false,
    speed: 1,
    cars: {
      x: [0, 1, 2],
      y: [0, 0, 0],
      psi: [0, 0, 0],
      v: [0, 0, 0],
      alive: [true, true, true],
      progress: [0, 0, 0],
      status: [0, 0, 0],
      action: [
        [0, 0],
        [0, 0],
        [0, 0],
      ],
    },
    selected: {
      index: selectedIndex,
      rays: [1, 2, 3, 4, 5],
      observation: [0, 0, 0, 0, 0, 0, 0],
      neuron: null,
      trace: null,
      explain: null,
    },
  };
}

const scenario: ScenarioMessage = {
  type: "scenario",
  track: "train_loop_01",
  policy: "random_networks",
  n_cars: 3,
  seed: 1,
  dt: 1 / 30,
  time_limit: 92,
  target_laps: 2,
  v_max: 30,
  wheelbase: 2.5,
  car_half_width: 1,
  ray_angles: [1.05, 0.52, 0, -0.52, -1.05],
  ray_range: 50,
};

let sent: { type: string; index?: number }[];

beforeEach(() => {
  useLive.setState({ ...initialLive });
  useSelection.getState().reset();
  sent = [];
  attachSender((text) => {
    sent.push(JSON.parse(text));
    return true;
  });
});

afterEach(() => attachSender(null));

describe("selectCar", () => {
  it("remembers the car and sends select_car once", () => {
    useSelection.getState().selectCar(2);
    expect(useSelection.getState().carId).toBe(2);
    expect(sent).toEqual([{ type: "select_car", index: 2 }]);
    useSelection.getState().selectCar(2); // already selected: nothing more to send
    expect(sent).toHaveLength(1);
  });

  it("keeps the selected neuron, so the same neuron can be compared across cars", () => {
    useSelection.getState().toggleNeuron({ layer: 1, index: 4 }, LAYERS);
    useSelection.getState().selectCar(1);
    expect(useSelection.getState().neuron).toEqual({ layer: 1, index: 4 });
  });
});

describe("toggleNeuron and toggleConnection", () => {
  it("selects a neuron, and clicking it again clears it", () => {
    const { toggleNeuron } = useSelection.getState();
    toggleNeuron({ layer: 2, index: 7 }, LAYERS);
    expect(useSelection.getState().neuron).toEqual({ layer: 2, index: 7 });
    expect(useSelection.getState().output).toBeNull();
    toggleNeuron({ layer: 2, index: 7 }, LAYERS);
    expect(useSelection.getState().neuron).toBeNull();
  });

  it("an output neuron also selects that output (steer first, throttle second)", () => {
    useSelection.getState().toggleNeuron({ layer: 3, index: 0 }, LAYERS);
    expect(useSelection.getState().output).toBe("steer");
    useSelection.getState().toggleNeuron({ layer: 3, index: 1 }, LAYERS);
    expect(useSelection.getState().output).toBe("throttle");
    useSelection.getState().toggleNeuron({ layer: 3, index: 1 }, LAYERS);
    expect(useSelection.getState().output).toBeNull();
  });

  it("a neuron and a connection replace each other", () => {
    const s = useSelection.getState();
    s.toggleNeuron({ layer: 1, index: 0 }, LAYERS);
    s.toggleConnection({ layer: 0, from: 2, to: 5 });
    expect(useSelection.getState().neuron).toBeNull();
    expect(useSelection.getState().connection).toEqual({ layer: 0, from: 2, to: 5 });
    s.toggleNeuron({ layer: 3, index: 0 }, LAYERS);
    expect(useSelection.getState().connection).toBeNull();
    expect(useSelection.getState().neuron).toEqual({ layer: 3, index: 0 });
  });

  it("clicking the same connection again clears it", () => {
    const s = useSelection.getState();
    s.toggleConnection({ layer: 1, from: 0, to: 0 });
    s.toggleConnection({ layer: 1, from: 0, to: 0 });
    expect(useSelection.getState().connection).toBeNull();
  });
});

describe("following the server", () => {
  it("takes the selected car from each frame", () => {
    useLive.getState().receive(scenario);
    useLive.getState().receive(frame(2));
    expect(useSelection.getState().carId).toBe(2);
    useLive.getState().receive(frame(0, 2));
    expect(useSelection.getState().carId).toBe(0);
    expect(sent).toEqual([]); // following the server never sends a command back
  });

  it("forgets neuron and connection selections when a new scenario arrives", () => {
    useSelection.getState().toggleNeuron({ layer: 3, index: 0 }, LAYERS);
    useLive.getState().receive(scenario);
    expect(useSelection.getState().neuron).toBeNull();
    expect(useSelection.getState().output).toBeNull();
    useSelection.getState().toggleConnection({ layer: 0, from: 1, to: 1 });
    useLive.getState().receive({ ...scenario, seed: 2 });
    expect(useSelection.getState().connection).toBeNull();
  });

  it("keeps the selection while frames of the same scenario arrive", () => {
    useLive.getState().receive(scenario);
    useSelection.getState().toggleNeuron({ layer: 1, index: 3 }, LAYERS);
    useLive.getState().receive(frame(0));
    useLive.getState().receive(frame(0, 2));
    expect(useSelection.getState().neuron).toEqual({ layer: 1, index: 3 });
  });
});

describe("raysFor", () => {
  const network: Pick<SelectedCar, "neuron" | "trace"> = {
    neuron: null,
    trace: { activations: [[0, 0, 0, 0, 0, 0, 0]], weighted_sums: [] },
  };
  const none = { neuron: null, connection: null };

  it("an input neuron that is a ray highlights that ray", () => {
    expect(raysFor({ ...none, neuron: { layer: 0, index: 0 } }, network)).toEqual([0]);
    expect(raysFor({ ...none, neuron: { layer: 0, index: 4 } }, network)).toEqual([4]);
  });

  it("speed and heading error have no ray; hidden and output neurons highlight nothing", () => {
    expect(raysFor({ ...none, neuron: { layer: 0, index: 5 } }, network)).toEqual([]);
    expect(raysFor({ ...none, neuron: { layer: 0, index: 6 } }, network)).toEqual([]);
    expect(raysFor({ ...none, neuron: { layer: 2, index: 1 } }, network)).toEqual([]);
  });

  it("a weight leaving a ray input highlights that ray", () => {
    expect(raysFor({ ...none, connection: { layer: 0, from: 3, to: 9 } }, network)).toEqual([3]);
    expect(raysFor({ ...none, connection: { layer: 1, from: 3, to: 9 } }, network)).toEqual([]);
  });

  it("the neuron driver highlights the two rays its neuron reads while it is selected", () => {
    const neuronDriver: Pick<SelectedCar, "neuron" | "trace"> = {
      trace: null,
      neuron: {
        inputs: [0.1, 0.2],
        input_rays: [0, 4],
        weights: [1, 1],
        bias: 0,
        contributions: [0.1, 0.2],
        z: 0.3,
        output: 0.29,
        activation: "tanh",
        throttle: 0.15,
      },
    };
    expect(raysFor(none, neuronDriver)).toEqual([]);
    expect(raysFor({ ...none, neuron: { layer: 0, index: 0 } }, neuronDriver)).toEqual([0, 4]);
  });

  it("nothing is highlighted without a frame or without a selection", () => {
    expect(raysFor({ ...none, neuron: { layer: 0, index: 1 } }, null)).toEqual([]);
    expect(raysFor(none, network)).toEqual([]);
  });
});
