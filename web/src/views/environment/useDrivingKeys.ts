import { useEffect } from "react";

import { useLive } from "../../state/live";
import { DRIVING_KEYS, inputFromKeys } from "./keys";

/** True when the key event came from a control the user is typing in or choosing from. */
function fromFormControl(event: KeyboardEvent): boolean {
  const target = event.target;
  return (
    target instanceof HTMLElement &&
    (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA")
  );
}

/**
 * Drive the car with the keyboard while the calling component is on screen.
 *
 * Arrow keys or WASD: up/W accelerate, down/S brake, left/A steer left, right/D steer right.
 * The held keys become a steer and throttle, and the store sends a `key_state` message only when
 * they change. Letting go of every key (or leaving the window) stops the input.
 */
export function useDrivingKeys(): void {
  useEffect(() => {
    const held = new Set<string>();
    const publish = () => useLive.getState().setKeys(inputFromKeys(held));

    const onKeyDown = (event: KeyboardEvent) => {
      if (!DRIVING_KEYS.has(event.code) || fromFormControl(event)) return;
      event.preventDefault(); // arrow keys would otherwise scroll the page
      if (!held.has(event.code)) {
        held.add(event.code);
        publish();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (held.delete(event.code)) publish();
    };
    const onBlur = () => {
      held.clear();
      publish();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      held.clear();
      publish();
    };
  }, []);
}
