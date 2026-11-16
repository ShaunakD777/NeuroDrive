import { describe, expect, it } from "vitest";

import type { PopulationInfo, ScenarioMessage } from "../api/generated";
import { type CrumbInput, crumbsFor, shortRun } from "./Breadcrumb";

const SCENARIO: ScenarioMessage = {
  type: "scenario",
  track: "train_loop_01",
  policy: "random_networks",
  n_cars: 20,
  seed: 0,
  checkpoint: null,
  population: null,
  start_shift: null,
  car_half_width: 0.9,
  wheelbase: 2.5,
  dt: 1 / 30,
  ray_angles: [],
  ray_range: 50,
  target_laps: 2,
  time_limit: 92,
  trail_seconds: 3,
  v_max: 30,
};

const POPULATION = {
  run: "20261110-101500-ga-default",
  unit: 7,
  last_unit: 30,
  start: 2,
  n_starts: 10,
  ghost_car: 50,
  ghost_unit: 5,
} as PopulationInfo;

function input(changes: Partial<CrumbInput>): CrumbInput {
  return {
    scenario: SCENARIO,
    selected: 3,
    layers: 4,
    neuron: null,
    connection: null,
    viewStep: null,
    ...changes,
  };
}

/** "Label value" pairs, the way a screen reader reads each crumb (the label "Car" above "Car 4"). */
const read = (i: CrumbInput) => crumbsFor(i).map((c) => `${c.label} ${c.value}`);

describe("crumbsFor", () => {
  it("is empty before the first scenario", () => {
    expect(crumbsFor(input({ scenario: null }))).toEqual([]);
  });

  it("names the driver and track, then car, neuron and moment", () => {
    expect(read(input({}))).toEqual([
      "Driver Random networks",
      "Track train_loop_01",
      "Car Car 4",
      "Neuron none",
      "Moment now",
    ]);
    const crumbs = crumbsFor(input({}));
    expect(crumbs.map((c) => c.view)).toEqual([
      "environment",
      "environment",
      "environment",
      "network",
      "environment",
    ]);
    expect(crumbs.filter((c) => c.empty).map((c) => c.label)).toEqual(["Neuron", "Moment"]);
  });

  it("starts from the run and checkpoint for a checkpoint replay", () => {
    const scenario = {
      ...SCENARIO,
      policy: "checkpoint" as const,
      checkpoint: { run: "20261101-090000-random_search-s3", unit: 120 },
    };
    expect(read(input({ scenario })).slice(0, 3)).toEqual([
      "Run random_search-s3",
      "Checkpoint unit 120",
      "Track train_loop_01",
    ]);
  });

  it("starts from the run, generation and start for a GA race, and names the ghost", () => {
    const scenario = { ...SCENARIO, policy: "population" as const, population: POPULATION };
    expect(read(input({ scenario })).slice(0, 4)).toEqual([
      "Run ga-default",
      "Generation 7 of 30",
      "Start 3 of 10",
      "Car Car 4",
    ]);
    expect(read(input({ scenario, selected: 50 }))[3]).toBe("Car Ghost · gen 5 champion");
  });

  it("names a selected input, hidden neuron, output or weight", () => {
    expect(read(input({ neuron: { layer: 0, index: 6 } }))[3]).toBe("Neuron Heading error");
    expect(read(input({ neuron: { layer: 1, index: 5 } }))[3]).toBe("Neuron Hidden 1 · 6");
    expect(read(input({ neuron: { layer: 3, index: 1 } }))[3]).toBe("Neuron Throttle");
    expect(read(input({ connection: { layer: 1, from: 2, to: 0 } }))[3]).toBe(
      "Neuron Weight L2 3→1",
    );
  });

  it("shows the look-back step and no car before the first frame", () => {
    expect(read(input({ viewStep: 240 }))[4]).toBe("Moment step 240");
    const crumbs = crumbsFor(input({ selected: null }));
    expect(crumbs[2]).toMatchObject({ label: "Car", value: "none", empty: true });
  });
});

describe("shortRun", () => {
  it("drops the date and time prefix and leaves other ids alone", () => {
    expect(shortRun("20261110-101500-ga-default")).toBe("ga-default");
    expect(shortRun("baseline-expert")).toBe("baseline-expert");
  });
});
