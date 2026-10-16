// Keyboard driving: which keys mean what, and how held keys become a steer and a throttle.

import type { DriveInput } from "../../state/live";

const LEFT = ["ArrowLeft", "KeyA"];
const RIGHT = ["ArrowRight", "KeyD"];
const FORWARD = ["ArrowUp", "KeyW"];
const BACK = ["ArrowDown", "KeyS"];

/** Every key code that drives the car (used to stop the page scrolling on arrow keys). */
export const DRIVING_KEYS: ReadonlySet<string> = new Set([...LEFT, ...RIGHT, ...FORWARD, ...BACK]);

function anyHeld(pressed: ReadonlySet<string>, codes: string[]): boolean {
  return codes.some((code) => pressed.has(code));
}

/**
 * Turn the set of currently held key codes into steer and throttle, each -1, 0 or +1.
 *
 * Steer: right is +1 and left is -1 (the server's convention: positive steer turns right).
 * Throttle: forward is +1 (accelerate) and back is -1 (brake). Opposite keys cancel out.
 */
export function inputFromKeys(pressed: ReadonlySet<string>): DriveInput {
  const steer = (anyHeld(pressed, RIGHT) ? 1 : 0) - (anyHeld(pressed, LEFT) ? 1 : 0);
  const throttle = (anyHeld(pressed, FORWARD) ? 1 : 0) - (anyHeld(pressed, BACK) ? 1 : 0);
  return { steer, throttle };
}
