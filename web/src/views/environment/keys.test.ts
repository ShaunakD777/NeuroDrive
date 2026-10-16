import { describe, expect, it } from "vitest";

import { DRIVING_KEYS, inputFromKeys } from "./keys";
import { describeStatus, formatRayAngle, formatSpeed } from "./status";

const held = (...codes: string[]) => new Set(codes);

describe("inputFromKeys", () => {
  it("is zero with nothing held", () => {
    expect(inputFromKeys(held())).toEqual({ steer: 0, throttle: 0 });
  });

  it("maps arrows and WASD: up accelerates, down brakes, right is +1 steer, left is -1", () => {
    expect(inputFromKeys(held("ArrowUp"))).toEqual({ steer: 0, throttle: 1 });
    expect(inputFromKeys(held("KeyW"))).toEqual({ steer: 0, throttle: 1 });
    expect(inputFromKeys(held("ArrowDown"))).toEqual({ steer: 0, throttle: -1 });
    expect(inputFromKeys(held("KeyS"))).toEqual({ steer: 0, throttle: -1 });
    expect(inputFromKeys(held("ArrowRight"))).toEqual({ steer: 1, throttle: 0 });
    expect(inputFromKeys(held("KeyD"))).toEqual({ steer: 1, throttle: 0 });
    expect(inputFromKeys(held("ArrowLeft"))).toEqual({ steer: -1, throttle: 0 });
    expect(inputFromKeys(held("KeyA"))).toEqual({ steer: -1, throttle: 0 });
  });

  it("combines steering and throttle", () => {
    expect(inputFromKeys(held("ArrowUp", "ArrowLeft"))).toEqual({ steer: -1, throttle: 1 });
  });

  it("lets opposite keys cancel, and two keys for one direction count once", () => {
    expect(inputFromKeys(held("ArrowLeft", "ArrowRight"))).toEqual({ steer: 0, throttle: 0 });
    expect(inputFromKeys(held("ArrowUp", "ArrowDown"))).toEqual({ steer: 0, throttle: 0 });
    expect(inputFromKeys(held("ArrowUp", "KeyW"))).toEqual({ steer: 0, throttle: 1 });
  });

  it("ignores keys that do not drive", () => {
    expect(inputFromKeys(held("Space", "KeyQ"))).toEqual({ steer: 0, throttle: 0 });
    expect(DRIVING_KEYS.has("Space")).toBe(false);
    expect(DRIVING_KEYS.has("ArrowUp")).toBe(true);
  });
});

describe("status words", () => {
  it("has no reason while running and a reason for every ending", () => {
    expect(describeStatus(0)).toMatchObject({ label: "Running", reason: null });
    expect(describeStatus(1).reason).toContain("left the road");
    expect(describeStatus(2).kind).toBe("good");
    expect(describeStatus(3).reason).toContain("stopped making progress");
    expect(describeStatus(4).label).toBe("Time limit");
    expect(describeStatus(9).label).toBe("Status 9");
  });

  it("formats speed in m/s and km/h", () => {
    expect(formatSpeed(10)).toBe("10.0 m/s (36 km/h)");
    expect(formatSpeed(0)).toBe("0.0 m/s (0 km/h)");
  });

  it("formats ray angles with a signed number of degrees", () => {
    expect(formatRayAngle(Math.PI / 3)).toBe("+60°");
    expect(formatRayAngle(0)).toBe("0°");
    expect(formatRayAngle(-Math.PI / 6)).toBe("−30°");
  });
});
