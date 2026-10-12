import { beforeEach, describe, expect, it } from "vitest";

import { initialSelection, useSelection } from "./selection";

describe("selection store", () => {
  beforeEach(() => {
    useSelection.getState().reset();
  });

  it("starts with nothing selected", () => {
    const state = useSelection.getState();
    expect(state.mode).toBe("live");
    expect(state.runId).toBeNull();
    expect(state.unit).toBeNull();
    expect(state.carId).toBeNull();
    expect(state.neuron).toBeNull();
    expect(state.connection).toBeNull();
    expect(state.step).toBe(0);
    expect(state.output).toBeNull();
    expect(state.whatIf).toEqual({});
  });

  it("select changes only the named fields", () => {
    useSelection.getState().select({ carId: 3, output: "steer" });
    const state = useSelection.getState();
    expect(state.carId).toBe(3);
    expect(state.output).toBe("steer");
    expect(state.mode).toBe(initialSelection.mode);
    expect(state.step).toBe(0);
  });

  it("reset clears everything, including what-if overrides", () => {
    useSelection.getState().select({
      mode: "replay",
      runId: "run-1",
      neuron: { layer: 1, index: 2 },
      whatIf: { 0: 0.5 },
    });
    useSelection.getState().reset();
    const state = useSelection.getState();
    expect(state.mode).toBe("live");
    expect(state.runId).toBeNull();
    expect(state.neuron).toBeNull();
    expect(state.whatIf).toEqual({});
  });
});
