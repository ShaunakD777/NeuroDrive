import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { FrameMessage, ScenarioMessage } from "../api/generated";
import { attachSender, initialLive, useLive } from "./live";

function frame(step: number, playing = false): FrameMessage {
  return {
    type: "frame",
    step,
    time: step / 30,
    playing,
    speed: 1,
    cars: {
      x: [0],
      y: [0],
      psi: [0],
      v: [0],
      alive: [true],
      progress: [0],
      status: [0],
      action: [[0, 0]],
    },
    selected: { index: 0, rays: [1, 2, 3, 4, 5], observation: [0, 0, 0, 0, 0, 0, 0] },
  };
}

const scenario: ScenarioMessage = {
  type: "scenario",
  track: "train_loop_01",
  policy: "keyboard",
  n_cars: 1,
  seed: null,
  dt: 1 / 30,
  time_limit: 90,
  target_laps: 2,
  v_max: 30,
  wheelbase: 2.5,
  car_half_width: 1,
  ray_angles: [1.05, 0.52, 0, -0.52, -1.05],
  ray_range: 50,
};

let sent: unknown[];

beforeEach(() => {
  useLive.setState({ ...initialLive });
  sent = [];
  attachSender((text) => {
    sent.push(JSON.parse(text));
    return true;
  });
});

afterEach(() => attachSender(null));

describe("live store: incoming messages", () => {
  it("keeps the scenario and the latest frame", () => {
    useLive.getState().receive(scenario);
    useLive.getState().receive(frame(0));
    useLive.getState().receive(frame(1));
    const state = useLive.getState();
    expect(state.scenario?.track).toBe("train_loop_01");
    expect(state.frame?.step).toBe(1);
  });

  it("ignores hello, unknown and malformed messages", () => {
    useLive.getState().receive({ type: "hello", version: "0", api: "v1.2" });
    useLive.getState().receive({ type: "nonsense" });
    useLive.getState().receive("text");
    useLive.getState().receive(null);
    expect(useLive.getState().frame).toBeNull();
    expect(useLive.getState().scenario).toBeNull();
  });

  it("remembers the episode ending until a fresh start (frame at step 0) clears it", () => {
    useLive.getState().receive(frame(60));
    useLive.getState().receive({
      type: "episode_end",
      step: 60,
      time: 2,
      status: ["crashed"],
      progress: [10],
      laps: [0],
    });
    expect(useLive.getState().episodeEnd?.status).toEqual(["crashed"]);
    useLive.getState().receive(frame(61));
    expect(useLive.getState().episodeEnd).not.toBeNull();
    useLive.getState().receive(frame(0));
    expect(useLive.getState().episodeEnd).toBeNull();
  });

  it("stores server errors, and a new scenario clears them along with the old frame", () => {
    useLive.getState().receive({ type: "error", message: "unknown track 'x'" });
    expect(useLive.getState().error).toBe("unknown track 'x'");
    useLive.getState().receive(frame(5));
    useLive.getState().receive(scenario);
    expect(useLive.getState().error).toBeNull();
    expect(useLive.getState().frame).toBeNull();
  });
});

describe("live store: commands", () => {
  it("loadTrack remembers the track and sends load_scenario with a keyboard car", () => {
    useLive.getState().loadTrack("train_loop_01");
    expect(useLive.getState().wantedTrack).toBe("train_loop_01");
    expect(sent).toEqual([
      { type: "load_scenario", track: "train_loop_01", policy: "keyboard", n_cars: 1, seed: null },
    ]);
  });

  it("sends key_state only when the input changes", () => {
    const { setKeys } = useLive.getState();
    setKeys({ steer: 0, throttle: 0 }); // nothing held and nothing sent before: no message
    setKeys({ steer: 0, throttle: 1 });
    setKeys({ steer: 0, throttle: 1 }); // unchanged: no message
    setKeys({ steer: -1, throttle: 1 });
    setKeys({ steer: 0, throttle: 0 });
    expect(sent).toEqual([
      { type: "key_state", steer: 0, throttle: 1 },
      { type: "key_state", steer: -1, throttle: 1 },
      { type: "key_state", steer: 0, throttle: 0 },
    ]);
  });

  it("does not remember keys it could not send, so the next change is retried", () => {
    attachSender(null);
    useLive.getState().setKeys({ steer: 1, throttle: 0 });
    expect(useLive.getState().keys).toEqual({ steer: 0, throttle: 0 });
    attachSender((text) => {
      sent.push(JSON.parse(text));
      return true;
    });
    useLive.getState().setKeys({ steer: 1, throttle: 0 });
    expect(sent).toEqual([{ type: "key_state", steer: 1, throttle: 0 }]);
  });

  it("forgets the sent keys when a new scenario arrives (the server's keyboard resets)", () => {
    useLive.getState().setKeys({ steer: 1, throttle: 1 });
    useLive.getState().receive(scenario);
    expect(useLive.getState().keys).toEqual({ steer: 0, throttle: 0 });
  });

  it("send reports failure when there is no socket", () => {
    attachSender(null);
    expect(useLive.getState().send({ type: "play" })).toBe(false);
  });
});
